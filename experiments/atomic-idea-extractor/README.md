# Prototype to extract atomic ideas from voice notes (or anything)

Look at config.py. Right now it expects a transcript.txt which is your raw note and the extraced structured JSON from this.

Use `voice_note_extractor.py` with an OpenAI key to extract atomic ideas into JSON.

Then use `streamlit run app/app.py` to launch the UI and click distill.

(right now the distill button just displays from the JSON data, doesn't actually call any Python functions to do the extraction in realtime.)