import streamlit as st
import pandas as pd

def render_archive_view(user_manager):
    """
    Display archived (removed) users with option to restore them.
    """
    archived = user_manager.get_archived_users()
    st.subheader("Archived Users")

    if not archived:
        st.info("No users have been archived yet.")
        return

    df = pd.DataFrame(archived)
    df = df[["user_id", "name", "subscription", "account_type", "capital", "status", "created_on", "removed_on"]]

    # Display each archived user row with a restore button
    for _, row in df.iterrows():
        cols = st.columns([2, 2, 2, 2, 2, 2, 3])
        with cols[0]:
            st.markdown(f"**{row['user_id']}**")
        with cols[1]:
            st.markdown(row["name"])
        with cols[2]:
            st.markdown(row["subscription"])
        with cols[3]:
            st.markdown(row["account_type"])
        with cols[4]:
            st.markdown(str(row["capital"]))
        with cols[5]:
            st.markdown(f"Removed on {row['removed_on']}")
        with cols[6]:
            if st.button("♻️ Restore", key=f"restore_{row['user_id']}"):
                success = user_manager.restore_user(row["user_id"])
                if success:
                    st.success(f"User {row['name']} restored successfully!")
                    st.experimental_rerun()
