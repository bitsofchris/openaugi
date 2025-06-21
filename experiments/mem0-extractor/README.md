# Mem0 Extractor - Streamlit App

A Streamlit application that demonstrates the use of the **Mem0 Open Source** library for extracting meaningful memories from large text files using OpenAI.

## Features

- 🔑 **Secure OpenAI API key management** with session state and environment variable cleanup
- 📄 **File upload support** for .txt and .md files
- 🧠 **Memory extraction** using Mem0 Open Source library with OpenAI
- 💭 **Display of extracted memories** with metadata and JSON view
- 🎨 **Modern light theme** with custom styling
- 📱 **Responsive layout** with expandable memory sections
- 🛡️ **Security features** including key masking and clear functionality

## Setup

1. **Install dependencies:**
   ```bash
   cd experiments/mem0-extractor
   pip install -r requirements.txt
   ```

2. **Set up your OpenAI API key:**
   - **Option 1**: Enter it directly in the app's sidebar (recommended)
   - **Option 2**: Set it as an environment variable: `export OPENAI_API_KEY="your-api-key-here"`

## Usage

1. **Launch the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

2. **Using the app:**
   - **Enter your OpenAI API key** in the sidebar (it will be masked for security)
   - **Upload a .txt or .md file** using the file uploader
   - **Click "Extract Memories"** to process the file with Mem0
   - **View the extracted memories** in expandable sections
   - **Use the "Clear API Key" button** to remove your key immediately

## File Structure

```
mem0-extractor/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── .streamlit/
    └── config.toml       # Streamlit configuration and theme
```

## Implementation Details

### Memory Extraction Process
1. **File Processing**: Text content is extracted from uploaded files
2. **Mem0 Integration**: Uses `Memory()` class from Mem0 Open Source
3. **Memory Addition**: Calls `m.add()` with user messages
4. **Memory Retrieval**: Uses `m.get_all()` to fetch all memories
5. **Display**: Shows memories in expandable sections with metadata

### Key Components
- **Mem0 Open Source**: `from mem0 import Memory`
- **OpenAI Integration**: Uses OpenAI API for memory extraction
- **Session Management**: Secure API key handling with Streamlit session state
- **Error Handling**: Comprehensive error handling with user feedback

## Requirements

- Python 3.8+
- OpenAI API key
- mem0ai library (Mem0 Open Source)
- Streamlit

## Troubleshooting

### Common Issues
1. **"No memories extracted"**: 
   - Check if your OpenAI API key is valid
   - Ensure you have sufficient OpenAI credits
   - Try with a different text file

2. **API Key errors**:
   - Use the "Clear API Key" button and re-enter
   - Check the masked key display for verification
   - Ensure no extra spaces in the key

3. **File upload issues**:
   - Only .txt and .md files are supported
   - Check file size and encoding

## License

This project is part of the OpenAugi experiments.
