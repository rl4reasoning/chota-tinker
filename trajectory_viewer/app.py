"""
Trajectory Viewer - A Streamlit app to browse multi-turn coding trajectories.

Usage:
    streamlit run trajectory_viewer/app.py
"""

import json
import streamlit as st
import pandas as pd

# Page config
st.set_page_config(
    page_title="Trajectory Viewer",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner="Loading dataset...")
def load_hf_dataset(dataset_name: str):
    """Load a dataset from HuggingFace Hub."""
    from datasets import load_dataset
    ds = load_dataset(dataset_name, split="train")
    return ds.to_pandas()


def parse_messages(messages_str: str) -> list[dict]:
    """Parse the messages JSON string into a list of dicts."""
    if isinstance(messages_str, list):
        return messages_str
    try:
        return json.loads(messages_str)
    except (json.JSONDecodeError, TypeError):
        return []


def parse_interactions(interactions_str: str) -> list[dict]:
    """Parse the interactions JSON string into a list of dicts."""
    if isinstance(interactions_str, list):
        return interactions_str
    try:
        return json.loads(interactions_str)
    except (json.JSONDecodeError, TypeError):
        return []


def render_message_content(content: str):
    """Render message content as plain text."""
    st.text(content)


def render_conversation(messages: list[dict]):
    """Render the conversation in chat-style format using plain text."""
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        
        # Handle different content types (Harmony format uses objects)
        if isinstance(content, dict):
            # Convert dict content to string representation
            content = json.dumps(content, indent=2)
        elif not isinstance(content, str):
            content = str(content)
        
        if role == "system":
            with st.expander("System Message", expanded=False):
                st.text(content)
        elif role == "developer":
            with st.expander("Developer Instructions", expanded=False):
                st.text(content)
        elif role == "user":
            with st.chat_message("user"):
                render_message_content(content)
        elif role == "assistant":
            with st.chat_message("assistant"):
                render_message_content(content)
        else:
            st.text(f"{role.title()}: {content}")


def render_interactions(interactions: list[dict]):
    """Render the code interaction history."""
    if not interactions:
        st.info("No code interactions recorded.")
        return
    
    for i, interaction in enumerate(interactions):
        with st.expander(f"Interaction {i + 1}", expanded=False):
            code = interaction.get("code", "")
            output = interaction.get("output", "")
            
            st.text("Code:")
            st.text(code)
            st.text("")
            st.text("Output:")
            st.text(output)


def main():
    st.title("Trajectory Viewer")
    st.text("Browse and analyze multi-turn coding trajectories from HuggingFace datasets.")
    
    # Sidebar
    with st.sidebar:
        st.header("Dataset")
        
        # Dataset input
        default_dataset = "bicycleman15/gpt_mt_8x16kx20"
        dataset_name = st.text_input(
            "HuggingFace Dataset",
            value=default_dataset,
            help="Enter a HuggingFace dataset path (e.g., username/dataset-name)"
        )
        
        load_button = st.button("Load Dataset", type="primary", use_container_width=True)
    
    # Initialize session state
    if "df" not in st.session_state:
        st.session_state.df = None
        st.session_state.loaded_dataset = None
    
    # Load dataset on button click or if already loaded
    if load_button or (st.session_state.loaded_dataset == dataset_name and st.session_state.df is not None):
        if load_button or st.session_state.df is None:
            try:
                st.session_state.df = load_hf_dataset(dataset_name)
                st.session_state.loaded_dataset = dataset_name
            except Exception as e:
                st.error(f"Failed to load dataset: {e}")
                return
    
    if st.session_state.df is None:
        st.info("Enter a HuggingFace dataset path and click 'Load Dataset' to begin.")
        st.text("Example datasets:")
        st.text("  - bicycleman15/gpt_mt_8x16kx20 - GPT multi-turn trajectories")
        return
    
    df = st.session_state.df
    
    # Sidebar filters
    with st.sidebar:
        st.divider()
        st.header("Filters")
        
        # Problem ID filter
        problem_ids = sorted(df["problem_id"].unique())
        selected_problem = st.selectbox(
            "Problem ID",
            options=["All"] + list(problem_ids),
            index=0,
        )
        
        # Filter dataframe by problem
        if selected_problem != "All":
            filtered_df = df[df["problem_id"] == selected_problem]
        else:
            filtered_df = df
        
        # Trajectory ID filter
        trajectory_ids = sorted(filtered_df["trajectory_id"].unique())
        selected_trajectory = st.selectbox(
            "Trajectory ID",
            options=["All"] + list(trajectory_ids),
            index=0,
        )
        
        if selected_trajectory != "All":
            filtered_df = filtered_df[filtered_df["trajectory_id"] == selected_trajectory]
        
        # Success filter
        success_filter = st.radio(
            "Success Status",
            options=["All", "Successful", "Failed"],
            horizontal=True,
        )
        
        if success_filter == "Successful":
            filtered_df = filtered_df[filtered_df["is_successful"] == True]
        elif success_filter == "Failed":
            filtered_df = filtered_df[filtered_df["is_successful"] == False]
        
        # Turns filter
        if "num_turns" in filtered_df.columns:
            min_turns = int(filtered_df["num_turns"].min())
            max_turns = int(filtered_df["num_turns"].max())
            if min_turns < max_turns:
                turns_range = st.slider(
                    "Number of Turns",
                    min_value=min_turns,
                    max_value=max_turns,
                    value=(min_turns, max_turns),
                )
                filtered_df = filtered_df[
                    (filtered_df["num_turns"] >= turns_range[0]) & 
                    (filtered_df["num_turns"] <= turns_range[1])
                ]
        
        st.divider()
        st.header("Statistics")
        
        total = len(df)
        successful = df["is_successful"].sum() if "is_successful" in df.columns else 0
        pass_rate = successful / total * 100 if total > 0 else 0
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total", total)
        with col2:
            st.metric("Filtered", len(filtered_df))
        
        st.metric("Pass Rate", f"{pass_rate:.1f}%")
        
        if "num_turns" in df.columns:
            avg_turns = df["num_turns"].mean()
            st.metric("Avg Turns", f"{avg_turns:.1f}")
    
    # Main content
    if len(filtered_df) == 0:
        st.warning("No trajectories match the current filters.")
        return
    
    # Show trajectory selector if multiple
    if len(filtered_df) > 1:
        st.text(f"Showing {len(filtered_df)} trajectories")
        
        # Create a summary table
        summary_cols = ["problem_id", "trajectory_id", "is_successful", "final_reward", "num_turns"]
        available_cols = [c for c in summary_cols if c in filtered_df.columns]
        
        # Add selection capability
        selected_idx = st.selectbox(
            "Select trajectory to view:",
            options=range(len(filtered_df)),
            format_func=lambda i: f"Problem {filtered_df.iloc[i]['problem_id']} | Traj {filtered_df.iloc[i]['trajectory_id']} | {'✅' if filtered_df.iloc[i].get('is_successful', False) else '❌'} | Reward: {filtered_df.iloc[i].get('final_reward', 'N/A')}"
        )
        
        row = filtered_df.iloc[selected_idx]
    else:
        row = filtered_df.iloc[0]
    
    # Display selected trajectory
    st.divider()
    
    # Header with metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Problem ID", row.get("problem_id", "N/A"))
    with col2:
        st.metric("Trajectory ID", row.get("trajectory_id", "N/A"))
    with col3:
        is_successful = row.get("is_successful", False)
        st.metric("Status", "✅ Success" if is_successful else "❌ Failed")
    with col4:
        reward = row.get("final_reward", 0)
        st.metric("Reward", f"{reward:.2f}" if isinstance(reward, float) else str(reward))
    
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("Turns", row.get("num_turns", "N/A"))
    with col6:
        st.metric("Terminated", "Yes" if row.get("terminated", False) else "No")
    with col7:
        st.metric("Truncated", "Yes" if row.get("truncated", False) else "No")
    with col8:
        interaction_timeouts = row.get("interaction_timeout_count", 0)
        st.metric("Interaction Timeouts", interaction_timeouts)
    
    st.divider()
    
    # Question section
    with st.expander("Problem Statement", expanded=False):
        question = row.get("question", "No question available")
        st.text(question)
    
    # Tabs for conversation and interactions
    tab1, tab2, tab3 = st.tabs(["Conversation", "Code Interactions", "Raw Data"])
    
    with tab1:
        messages = parse_messages(row.get("messages", "[]"))
        if messages:
            render_conversation(messages)
        else:
            st.info("No messages available.")
    
    with tab2:
        interactions = parse_interactions(row.get("interactions", "[]"))
        render_interactions(interactions)
    
    with tab3:
        # Show raw JSON data
        st.json({
            "problem_id": row.get("problem_id"),
            "trajectory_id": row.get("trajectory_id"),
            "question": row.get("question", "")[:200] + "..." if len(row.get("question", "")) > 200 else row.get("question", ""),
            "num_turns": row.get("num_turns"),
            "final_reward": row.get("final_reward"),
            "is_successful": row.get("is_successful"),
            "terminated": row.get("terminated"),
            "truncated": row.get("truncated"),
            "interaction_timeout_count": row.get("interaction_timeout_count"),
            "eval_timeout_count": row.get("eval_timeout_count"),
        })


if __name__ == "__main__":
    main()
