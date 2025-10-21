import streamlit as st
import pandas as pd

def render_user_table(user_manager):
    """
    Render a user management table with inline action buttons.
    """
    users = user_manager.get_users()
    if not users:
        st.info("No users registered.")
        return
    
    df = pd.DataFrame(users)

    st.subheader("User Management")

    # Display each user row with actions
    for _, row in df.iterrows():
        col1, col2, col3, col4, col5, col6, col7, col8 = st.columns([1, 2, 2, 2, 2, 2, 2, 4])

        with col1:
            st.markdown(f"**{row['user_id']}**")
        with col2:
            st.markdown(row["name"])
        with col3:
            st.markdown(row["subscription"])
        with col4:
            st.markdown(row["account_type"])
        with col5:
            status_color = "green" if row["status"] == "Active" else "gray"
            st.markdown(f"<span style='color:{status_color}'>{row['status']}</span>", unsafe_allow_html=True)
        with col6:
            st.markdown(str(row["capital"]))
        with col7:
            st.markdown(row["created_on"])

        # Inline action buttons
        with col8:
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("❌ Remove", key=f"remove_{row['user_id']}"):
                    user_manager.remove_user(row["user_id"])
                    st.success(f"User {row['name']} removed")
                    st.experimental_rerun()
            with c2:
                if st.button("✏️ Edit", key=f"edit_{row['user_id']}"):
                    st.session_state["edit_user_id"] = row["user_id"]
            with c3:
                toggle_label = "Deactivate" if row["status"] == "Active" else "Activate"
                if st.button(f"🔄 {toggle_label}", key=f"status_{row['user_id']}"):
                    new_status = "Inactive" if row["status"] == "Active" else "Active"
                    user_manager.set_user_status(row["user_id"], new_status)
                    st.success(f"User {row['name']} set to {new_status}")
                    st.experimental_rerun()

    # Edit user form (if triggered)
    if "edit_user_id" in st.session_state:
        user_id = st.session_state["edit_user_id"]
        user = next((u for u in users if u["user_id"] == user_id), None)
        if user:
            st.subheader(f"Edit User: {user['name']}")
            with st.form(f"edit_form_{user_id}"):
                name = st.text_input("Name", user["name"])
                subscription = st.selectbox(
                    "Subscription", 
                    ["Free", "Basic", "Pro", "Premium"],
                    index=["Free", "Basic", "Pro", "Premium"].index(user["subscription"])
                )
                account_type = st.selectbox("Account Type", ["Demo", "Live"], index=0 if user["account_type"]=="Demo" else 1)
                capital = st.number_input("Capital", min_value=0.0, step=100.0, value=float(user["capital"]))
                session = st.selectbox(
                    "Session", ["Tokyo", "London", "New York"],
                    index=0 if user["session"]=="Tokyo" else (1 if user["session"]=="London" else 2)
                )
                telegram_id = st.text_input("Telegram ID", user.get("telegram_id", ""))

                submit = st.form_submit_button("Save Changes")
                if submit:
                    user_manager.update_user(
                        user_id,
                        name=name,
                        subscription=subscription,
                        account_type=account_type,
                        capital=capital,
                        session=session,
                        telegram_id=telegram_id
                    )
                    st.success(f"User {name} updated successfully!")
                    del st.session_state["edit_user_id"]
                    st.experimental_rerun()
