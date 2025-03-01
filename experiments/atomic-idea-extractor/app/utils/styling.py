import streamlit as st


def apply_custom_css():
    custom_css = """
    <style>
    /* Target the expander's content area to give it a card-like appearance */
    [data-testid="stExpander"] > div:nth-child(2) {
      background-color: #ffffff;
      border-radius: 8px;
      padding: 16px;
      box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.1);
      margin-bottom: 16px;
      transition: transform 0.3s ease;
    }
    [data-testid="stExpander"] > div:nth-child(2):hover {
      transform: scale(1.02);
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
