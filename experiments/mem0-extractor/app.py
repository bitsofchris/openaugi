import streamlit as st
import os
from mem0 import Memory

### Hide the top decoration bar
hide_decoration_bar_style = """
    <style>
        header {visibility: hidden;}
    </style>
"""
st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)

### Page configuration
st.set_page_config(page_title="Mem0 Extractor", page_icon="🧠", layout="wide")

### Sidebar for API key input
with st.sidebar:
    st.header("🔑 Configuration")

    # Use session state to manage API key securely
    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = ""

    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=st.session_state.openai_api_key,
        help="Enter your OpenAI API key to enable memory extraction",
        key="api_key_input",
    )

    # Update session state when user changes the key
    if api_key != st.session_state.openai_api_key:
        st.session_state.openai_api_key = api_key

    if api_key:
        st.success("✅ API key configured")
        # Show a masked version for verification
        masked_key = (
            api_key[:4] + "*" * (len(api_key) - 8) + api_key[-4:]
            if len(api_key) > 8
            else "*" * len(api_key)
        )
        st.info(f"Using key: {masked_key}")
    else:
        st.warning("⚠️ Please enter your OpenAI API key")

    # Add a clear button for security
    if st.button("🗑️ Clear API Key"):
        st.session_state.openai_api_key = ""
        st.rerun()

### Main content
st.title("🧠 Mem0 Memory Extractor")
st.subheader("Extract meaningful memories from your text documents")

### File upload section
st.header("📄 Upload Document")
uploaded_file = st.file_uploader(
    "Choose a text file",
    type=["txt", "md"],
    help="Upload a .txt or .md file to extract memories from",
)

### Display file info if uploaded
if uploaded_file is not None:
    st.success(f"✅ File uploaded: {uploaded_file.name}")
    st.info(f"File size: {uploaded_file.size} bytes")

    with st.expander("📖 File Preview"):
        file_content = uploaded_file.read().decode("utf-8")
        st.text_area("File content:", file_content, height=200)
        uploaded_file.seek(0)  # <--- Reset file pointer for later use

### Extract button
st.header("🔍 Extract Memories")
extract_button = st.button(
    "🚀 Extract Memories",
    type="primary",
    disabled=not (uploaded_file is not None and api_key),
    help="Click to extract memories from the uploaded file",
)

### Memory display section
st.header("💭 Extracted Memories")

if extract_button:
    if uploaded_file is not None and api_key:
        with st.spinner("Extracting memories..."):
            try:
                ### Initialize Mem0 Open Source
                # Set the OpenAI API key as environment variable (temporary, process-scoped)
                original_key = os.environ.get("OPENAI_API_KEY", "")
                os.environ["OPENAI_API_KEY"] = api_key

                # Initialize the Memory class (open source version)
                m = Memory()

                ### Extract memories
                messages = [{"role": "user", "content": file_content}]
                m.add(
                    messages,
                    user_id="default_user",
                    metadata={"source": "streamlit_upload"},
                )

                st.success("✅ Memories extracted successfully!")

                ### Display extracted memories
                all_memories = m.get_all(user_id="default_user")

                with st.expander("All Memories (JSON)"):
                    st.write(all_memories)

                if (
                    all_memories
                    and "results" in all_memories
                    and all_memories["results"]
                ):
                    st.write(
                        f"**Total memories extracted:** {len(all_memories['results'])}"
                    )

                    for i, memory in enumerate(all_memories["results"], 1):
                        with st.expander(f"Memory {i}"):
                            st.write(f"**Memory:** {memory['memory']}")
                            st.write(f"**Created:** {memory['created_at']}")
                else:
                    st.info(
                        "No memories were extracted from this document. This might be because the content doesn't contain extractable memories or the extraction process didn't identify any meaningful information."
                    )

                # Clean up: restore original environment variable
                if original_key:
                    os.environ["OPENAI_API_KEY"] = original_key
                else:
                    os.environ.pop("OPENAI_API_KEY", None)

            except Exception as e:
                st.error(f"❌ Error during memory extraction: {str(e)}")
                st.info(
                    "💡 Make sure your OpenAI API key is valid and you have sufficient credits."
                )

                # Clean up on error too
                if original_key:
                    os.environ["OPENAI_API_KEY"] = original_key
                else:
                    os.environ.pop("OPENAI_API_KEY", None)
    else:
        st.error("❌ Please upload a file and provide an API key to extract memories.")

### Footer
st.markdown("---")
st.markdown("*Powered by Mem0 Open Source Library*")

# Security note
with st.expander("🔒 Security Information"):
    st.info(
        """
    **API Key Security:**
    - Your API key is stored only in Streamlit's session state (in-memory)
    - It's never saved to disk or logged
    - The key is cleared when you refresh the page or close the app
    - Use the 'Clear API Key' button to remove it immediately
    - Environment variables are restored after each operation
    """
    )
