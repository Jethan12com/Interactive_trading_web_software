import pandas as pd
import numpy as np
import pandas_ta as ta
import logging
import requests
import os
from datetime import datetime
from sklearn.ensemble import IsolationForest
import asyncio
import websockets
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class PatternDiscovery:
    def __init__(self, data_provider):
        self.data_provider = data_provider
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(filename='logs/pattern_discovery.log', level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s')
        self.xai_api_url = "wss://api.x.ai/v1/sentiment/stream"
        self.finnhub_api_url = "https://finnhub.io/api/v1/news-sentiment"
        self.xai_api_key = os.getenv('XAI_API_KEY')
        self.finnhub_api_key = os.getenv('FINNHUB_API_KEY')
        self.sentiment_weight_x = 0.6
        self.sentiment_weight_news = 0.4
        self.timeframe_weights = {'1H': 0.2, '4H': 0.3, 'D1': 0.5}
        self.indicator_weights = {'candlestick': 0.3, 'anomaly': 0.2, 'divergence': 0.2, 'macd': 0.15, 'bollinger': 0.15}
        self.iso_forest = IsolationForest(contamination=0.05, random_state=42)
        self.sentiment_cache = {}

    # -------------------- SENTIMENT --------------------
    async def fetch_x_sentiment_stream(self, pair, callback):
        try:
            async with websockets.connect(self.xai_api_url, extra_headers={'Authorization': f'Bearer {self.xai_api_key}'}) as ws:
                await ws.send(json.dumps({'pair': pair, 'subscribe': True}))
                self.logger.info(f"Connected to X sentiment stream for {pair}")
                while True:
                    message = await ws.recv()
                    data = json.loads(message)
                    sentiment_df = pd.DataFrame({
                        'timestamp': [pd.to_datetime(data['timestamp'])],
                        'sentiment_score': [data['sentiment_score']]
                    }).set_index('timestamp')
                    self.sentiment_cache[pair] = sentiment_df
                    callback(sentiment_df)
                    self.logger.info(f"Received real-time X sentiment for {pair}: {data['sentiment_score']}")
        except Exception as e:
            self.logger.error(f"X sentiment stream failed for {pair}: {e}")
            return pd.DataFrame()

    def fetch_news_sentiment(self, pair, start_date, end_date):
        try:
            headers = {'X-Finnhub-Token': self.finnhub_api_key}
            symbol = pair.split('/')[0]
            params = {'symbol': symbol, 'from': start_date, 'to': end_date}
            response = requests.get(self.finnhub_api_url, headers=headers, params=params)
            response.raise_for_status()
            sentiment_df = pd.DataFrame(response.json().get('sentiment', []))
            if not sentiment_df.empty:
                sentiment_df['timestamp'] = pd.to_datetime(sentiment_df['timestamp'])
                sentiment_df.set_index('timestamp', inplace=True)
                self.logger.info(f"Fetched {len(sentiment_df)} news sentiment for {pair}")
                return sentiment_df[['sentiment_score']]
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"News sentiment fetch failed for {pair}: {e}")
            return pd.DataFrame()

    def combine_sentiment(self, x_sentiment_df, news_sentiment_df):
        try:
            if x_sentiment_df.empty and news_sentiment_df.empty:
                return pd.DataFrame({'combined_sentiment': []})
            combined_df = x_sentiment_df.join(news_sentiment_df, how='outer', lsuffix='_x', rsuffix='_news').fillna(0)
            combined_df['combined_sentiment'] = (
                self.sentiment_weight_x * combined_df['sentiment_score_x'] +
                self.sentiment_weight_news * combined_df['sentiment_score_news']
            )
            return combined_df[['combined_sentiment']]
        except Exception as e:
            self.logger.error(f"Sentiment combination failed: {e}")
            return pd.DataFrame({'combined_sentiment': []})

    # -------------------- DATA RESAMPLING --------------------
    def resample_data(self, df, timeframe):
        try:
            agg_dict = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
            if timeframe in ['1H', '4H', 'D1']:
                return df.resample(timeframe[0]).agg(agg_dict).dropna()
            self.logger.warning(f"Unsupported timeframe: {timeframe}")
            return df
        except Exception as e:
            self.logger.error(f"Data resampling failed for {timeframe}: {e}")
            return df

    # -------------------- CANDLESTICKS --------------------
    def detect_candlestick_patterns(self, df, pair):
        try:
            if df.empty or len(df) < 5:
                return pd.DataFrame()
            df_1h, df_4h, df_d1 = [self.resample_data(df, tf) for tf in ['1H', '4H', 'D1']]
            signals = [self._detect_candlestick_single(df_tf, pair, tf) for df_tf, tf in zip([df_1h, df_4h, df_d1], ['1H','4H','D1'])]
            return self.combine_timeframe_signals(*signals, pair, 'candlestick')
        except Exception as e:
            self.logger.error(f"Candlestick detection failed for {pair}: {e}")
            return pd.DataFrame()

    def _detect_candlestick_single(self, df, pair, timeframe):
        try:
            candles = ta.cdl_pattern(df, name=["doji","hammer","engulfing"])
            signals = []
            for index, row in candles.iterrows():
                signal_value = confidence = 0.0
                if row['CDL_DOJI'] != 0:
                    signal_value, confidence = np.sign(row['CDL_DOJI']), 0.6
                elif row['CDL_HAMMER'] > 0:
                    signal_value, confidence = 1.0, 0.8
                elif row['CDL_ENGULFING'] != 0:
                    signal_value, confidence = np.sign(row['CDL_ENGULFING']), 0.85
                if confidence > 0:
                    signals.append({
                        'timestamp': index, 'pair': pair, 'signal':'candlestick',
                        'signal_value': signal_value, 'confidence': confidence, 'timeframe': timeframe,
                        'signal_magnitude': confidence
                    })
            return pd.DataFrame(signals)
        except Exception as e:
            self.logger.error(f"Candlestick detection failed for {pair} on {timeframe}: {e}")
            return pd.DataFrame()

    # -------------------- ANOMALIES --------------------
    def detect_anomalies(self, df, pair, features=['close','volume','combined_sentiment']):
        try:
            if df.empty or len(df) < 20:
                return pd.DataFrame()
            df = self.get_sentiment_feature(df, pair)
            df_1h, df_4h, df_d1 = [self.resample_data(df, tf) for tf in ['1H','4H','D1']]
            signals = [self._detect_anomalies_single(df_tf, pair, features, tf) for df_tf, tf in zip([df_1h,df_4h,df_d1],['1H','4H','D1'])]
            return self.combine_timeframe_signals(*signals, pair, 'anomaly')
        except Exception as e:
            self.logger.error(f"Anomaly detection failed for {pair}: {e}")
            return pd.DataFrame()

    def _detect_anomalies_single(self, df, pair, features, timeframe):
        try:
            data = df[features].dropna()
            if len(data) < 20: return pd.DataFrame()
            anomalies = self.iso_forest.fit_predict(data)
            scores = -self.iso_forest.score_samples(data)
            signals = [{
                'timestamp': index, 'pair': pair, 'signal':'anomaly',
                'signal_value': scores[i], 'confidence': min(scores[i],1.0), 'timeframe': timeframe,
                'signal_magnitude': min(scores[i],1.0)
            } for i, (index, anomaly) in enumerate(zip(data.index, anomalies)) if anomaly==-1]
            return pd.DataFrame(signals)
        except Exception as e:
            self.logger.error(f"Anomaly detection failed for {pair} on {timeframe}: {e}")
            return pd.DataFrame()

    # -------------------- DIVERGENCES --------------------
    def detect_divergences(self, df, pair, indicator='rsi'):
        try:
            if df.empty or len(df) < 14:
                return pd.DataFrame()
            df_1h, df_4h, df_d1 = [self.resample_data(df, tf) for tf in ['1H','4H','D1']]
            signals = [self._detect_divergences_single(df_tf, pair, indicator, tf) for df_tf, tf in zip([df_1h,df_4h,df_d1],['1H','4H','D1'])]
            return self.combine_timeframe_signals(*signals, pair, 'divergence')
        except Exception as e:
            self.logger.error(f"Divergence detection failed for {pair}: {e}")
            return pd.DataFrame()

    def _detect_divergences_single(self, df, pair, indicator, timeframe):
        try:
            rsi = ta.rsi(df['close'], length=14)
            signals = []
            for i in range(2,len(df)):
                price_diff = df['close'].iloc[i] - df['close'].iloc[i-2]
                rsi_diff = rsi.iloc[i] - rsi.iloc[i-2]
                signal_value = confidence = 0.0
                if price_diff>0 and rsi_diff<0:
                    signal_value, confidence = -1.0, 0.75
                elif price_diff<0 and rsi_diff>0:
                    signal_value, confidence = 1.0, 0.75
                if confidence>0:
                    signals.append({
                        'timestamp': df.index[i], 'pair': pair, 'signal':'divergence',
                        'signal_value': signal_value, 'confidence': confidence, 'timeframe': timeframe,
                        'signal_magnitude': confidence
                    })
            return pd.DataFrame(signals)
        except Exception as e:
            self.logger.error(f"Divergence detection failed for {pair} on {timeframe}: {e}")
            return pd.DataFrame()

    # -------------------- MACD --------------------
    def detect_macd(self, df, pair):
        try:
            if df.empty or len(df)<26:
                return pd.DataFrame()
            df_1h, df_4h, df_d1 = [self.resample_data(df, tf) for tf in ['1H','4H','D1']]
            signals = [self._detect_macd_single(df_tf, pair, tf) for df_tf, tf in zip([df_1h,df_4h,df_d1],['1H','4H','D1'])]
            return self.combine_timeframe_signals(*signals, pair, 'macd')
        except Exception as e:
            self.logger.error(f"MACD detection failed for {pair}: {e}")
            return pd.DataFrame()

    def _detect_macd_single(self, df, pair, timeframe):
        try:
            macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
            signals=[]
            for i in range(1,len(df)):
                if macd['MACD_12_26_9'].iloc[i] > macd['MACDs_12_26_9'].iloc[i] and macd['MACD_12_26_9'].iloc[i-1] <= macd['MACDs_12_26_9'].iloc[i-1]:
                    signals.append({'timestamp':df.index[i],'pair':pair,'signal':'macd','signal_value':1.0,'confidence':0.8,'timeframe':timeframe,'signal_magnitude':0.8})
                elif macd['MACD_12_26_9'].iloc[i] < macd['MACDs_12_26_9'].iloc[i] and macd['MACD_12_26_9'].iloc[i-1] >= macd['MACDs_12_26_9'].iloc[i-1]:
                    signals.append({'timestamp':df.index[i],'pair':pair,'signal':'macd','signal_value':-1.0,'confidence':0.8,'timeframe':timeframe,'signal_magnitude':0.8})
            return pd.DataFrame(signals)
        except Exception as e:
            self.logger.error(f"MACD detection failed for {pair} on {timeframe}: {e}")
            return pd.DataFrame()

    # -------------------- BOLLINGER --------------------
    def detect_bollinger(self, df, pair):
        try:
            if df.empty or len(df)<20:
                return pd.DataFrame()
            df_1h, df_4h, df_d1 = [self.resample_data(df, tf) for tf in ['1H','4H','D1']]
            signals=[self._detect_bollinger_single(df_tf, pair, tf) for df_tf, tf in zip([df_1h,df_4h,df_d1],['1H','4H','D1'])]
            return self.combine_timeframe_signals(*signals,pair,'bollinger')
        except Exception as e:
            self.logger.error(f"Bollinger Bands detection failed for {pair}: {e}")
            return pd.DataFrame()

    def _detect_bollinger_single(self, df, pair, timeframe):
        try:
            bb = ta.bbands(df['close'], length=20, std=2)
            signals=[]
            for i in range(1,len(df)):
                if df['close'].iloc[i]<bb['BBL_20_2.0'].iloc[i] and df['close'].iloc[i-1]>=bb['BBL_20_2.0'].iloc[i-1]:
                    signals.append({'timestamp':df.index[i],'pair':pair,'signal':'bollinger','signal_value':1.0,'confidence':0.85,'timeframe':timeframe,'signal_magnitude':0.85})
                elif df['close'].iloc[i]>bb['BBU_20_2.0'].iloc[i] and df['close'].iloc[i-1]<=bb['BBU_20_2.0'].iloc[i-1]:
                    signals.append({'timestamp':df.index[i],'pair':pair,'signal':'bollinger','signal_value':-1.0,'confidence':0.85,'timeframe':timeframe,'signal_magnitude':0.85})
            return pd.DataFrame(signals)
        except Exception as e:
            self.logger.error(f"Bollinger Bands detection failed for {pair} on {timeframe}: {e}")
            return pd.DataFrame()

    # -------------------- SENTIMENT FEATURE --------------------
    def get_sentiment_feature(self, df, pair, start_date='2024-01-01', end_date=None):
        try:
            if end_date is None:
                end_date = datetime.now().isoformat()
            x_sentiment_df = self.sentiment_cache.get(pair, pd.DataFrame())
            news_sentiment_df = self.fetch_news_sentiment(pair, start_date, end_date)
            combined_sentiment_df = self.combine_sentiment(x_sentiment_df, news_sentiment_df)
            df = df.join(combined_sentiment_df, how='left').fillna({'combined_sentiment':0.0})
            return df
        except Exception as e:
            self.logger.error(f"Sentiment feature failed for {pair}: {e}")
            df['combined_sentiment'] = 0.0
            return df

    # -------------------- COMBINE TIMEFRAME --------------------
    def combine_timeframe_signals(self, signals_1h, signals_4h, signals_d1, pair, signal_type):
        try:
            signals=[]
            timestamps = set(signals_1h.get('timestamp',[])).union(signals_4h.get('timestamp',[]), signals_d1.get('timestamp',[]))
            for ts in timestamps:
                confidence = signal_value = 0.0
                for signals_df, tf in [(signals_1h,'1H'),(signals_4h,'4H'),(signals_d1,'D1')]:
                    signal = signals_df[signals_df.get('timestamp')==ts] if not signals_df.empty else pd.DataFrame()
                    if not signal.empty:
                        confidence += self.timeframe_weights[tf]*signal['confidence'].iloc[0]
                        signal_value += self.timeframe_weights[tf]*signal['signal_value'].iloc[0]
                if confidence>0:
                    signals.append({'timestamp':ts,'pair':pair,'signal':signal_type,'signal_value':signal_value,'confidence':min(confidence,1.0),'signal_magnitude':signal_value})
            return pd.DataFrame(signals)
        except Exception as e:
            self.logger.error(f"Failed to combine {signal_type} signals for {pair}: {e}")
            return pd.DataFrame()

    # -------------------- COMBINE ALL SIGNALS --------------------
    def combine_all_signals(self, df, pair):
        try:
            signals_list = [self.detect_candlestick_patterns(df,pair), self.detect_anomalies(df,pair),
                            self.detect_divergences(df,pair), self.detect_macd(df,pair), self.detect_bollinger(df,pair)]
            df_combined = df.copy()
            for col in ['candlestick_signal','anomaly_signal','divergence_signal','macd_signal','bollinger_signal']:
                df_combined[col]=0.0
            for signal_df, signal_type in zip(signals_list, ['candlestick','anomaly','divergence','macd','bollinger']):
                for _,row in signal_df.iterrows():
                    df_combined.loc[row['timestamp'], f"{signal_type}_signal"]=row['signal_value']*self.indicator_weights[signal_type]
            df_combined['combined_signal_score']=df_combined[['candlestick_signal','anomaly_signal','divergence_signal','macd_signal','bollinger_signal']].sum(axis=1)
            return df_combined
        except Exception as e:
            self.logger.error(f"Signal combination failed for {pair}: {e}")
            return df

    # -------------------- MULTI-PANEL PLOT --------------------
    def plot_signals_multiplot(self, df_ohlcv, signals_df, pair, title=None, ma_periods=[20,50], bollinger=True, rsi_period=14):
        if df_ohlcv.empty or signals_df.empty: return
        df_plot = df_ohlcv.copy()
        df_plot.index=pd.to_datetime(df_plot.index)
        for period in ma_periods: df_plot[f'MA_{period}']=df_plot['close'].rolling(period).mean()
        if bollinger:
            df_plot['BB_Mid']=df_plot['close'].rolling(20).mean()
            df_plot['BB_Std']=df_plot['close'].rolling(20).std()
            df_plot['BB_Upper']=df_plot['BB_Mid']+2*df_plot['BB_Std']
            df_plot['BB_Lower']=df_plot['BB_Mid']-2*df_plot['BB_Std']
        df_plot['RSI']=ta.rsi(df_plot['close'], length=rsi_period)
        overbought_intensity=np.clip((df_plot['RSI']-70)/30,0,1)
        oversold_intensity=np.clip((30-df_plot['RSI'])/30,0,1)

        fig, axes=plt.subplots(6,1,figsize=(16,20), sharex=True, gridspec_kw={'height_ratios':[3,1,1,1,1,1]})
        plt.subplots_adjust(hspace=0.05)

        # Candlesticks
        for i in range(len(df_plot)):
            color='green' if df_plot['close'].iloc[i]>=df_plot['open'].iloc[i] else 'red'
            axes[0].vlines(df_plot.index[i],df_plot['low'].iloc[i],df_plot['high'].iloc[i],color=color)
            axes[0].vlines(df_plot.index[i],df_plot['open'].iloc[i],df_plot['close'].iloc[i],color=color,linewidth=4)
        for period in ma_periods: axes[0].plot(df_plot.index,df_plot[f'MA_{period}'],label=f'MA {period}')
        if bollinger:
            axes[0].plot(df_plot.index,df_plot['BB_Upper'],color='orange',linestyle='--',label='BB Upper')
            axes[0].plot(df_plot.index,df_plot['BB_Lower'],color='orange',linestyle='--',label='BB Lower')
            axes[0].plot(df_plot.index,df_plot['BB_Mid'],color='blue',linestyle=':',label='BB Mid')
        for i in range(len(df_plot)-1):
            if overbought_intensity.iloc[i]>0: axes[0].axvspan(df_plot.index[i],df_plot.index[i+1],color='red',alpha=0.1+0.4*overbought_intensity.iloc[i])
            if oversold_intensity.iloc[i]>0: axes[0].axvspan(df_plot.index[i],df_plot.index[i+1],color='green',alpha=0.1+0.4*oversold_intensity.iloc[i])
        bullish=signals_df[signals_df.get('signal_value',0)>0]
        bearish=signals_df[signals_df.get('signal_value',0)<0]
        axes[0].scatter(bullish.index,df_plot.loc[bullish.index,'close'],marker='^',color='lime',s=50*bullish.get('signal_magnitude',1.0),label='Bullish Signal')
        axes[0].scatter(bearish.index,df_plot.loc[bearish.index,'close'],marker='v',color='magenta',s=50*bearish.get('signal_magnitude',1.0),label='Bearish Signal')
        axes[0].set_ylabel('Price')
        axes[0].set_title(title or f"{pair} OHLC and Signals")
        axes[0].legend(loc='upper left'); axes[0].grid(True)

        for panel, col_name, ylabel in zip([axes[1],axes[2],axes[3],axes[4]],
                                           ['macd_signal','divergence_signal','bollinger_signal','anomaly_signal'],
                                           ['MACD Signal','Divergence','Bollinger','Anomaly']):
            if col_name in signals_df: panel.bar(signals_df.index,signals_df[col_name],color=['green' if v>0 else 'red' for v in signals_df[col_name]]); panel.set_ylabel(ylabel); panel.grid(True)

        if 'combined_signal_score' in signals_df:
            magnitudes = signals_df.get('signal_magnitude',pd.Series(1.0,index=signals_df.index))
            bar_heights = signals_df['combined_signal_score']*magnitudes
            colors = ['green' if v>0 else 'red' for v in signals_df['combined_signal_score']]
            axes[5].bar(signals_df.index,bar_heights,color=colors); axes[5].set_ylabel('Combined Score'); axes[5].grid(True)

        axes[5].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.xticks(rotation=45); plt.show()
        self.logger.info(f"Plotted multi-panel signals with magnitude-scaled combined scores for {pair}")