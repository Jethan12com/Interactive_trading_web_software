from flask import Flask, request
import stripe
from modules.user_management import UserManagement

app = Flask(__name__)
user_manager = UserManagement()

@app.route('/webhook', methods=['POST'])
def stripe_webhook():
    payload = request.data
    sig_header = request.headers.get('Stripe-Signature')
    event = None
    try:
        event = stripe.Webhook.construct_event(payload, sig_header, os.getenv('STRIPE_WEBHOOK_SECRET'))
    except Exception as e:
        return str(e), 400

    if event['type'] == 'invoice.payment_succeeded':
        customer_id = event['data']['object']['customer']
        users = user_manager.user_store.get_all_users()
        user = users[users['stripe_customer_id'] == customer_id]
        if not user.empty:
            user_manager.renew_subscription(user.iloc[0]['user_id'])

    return '', 200
