import streamlit as st
import streamlit.components.v1 as components
import base64

# Streamlit app title
st.title("Streamlit App with Background Music")

# Path to the local audio file
audio_file_path = "audio.mp3"  # Update this path as needed
icon_file_path = "icon.jpg"  # Update this path to the location of your icon file

# Function to read the audio file and encode it to base64
def get_base64_audio(file_path):
    with open(file_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode()

# Function to read the icon file and encode it to base64
def get_base64_icon(file_path):
    with open(file_path, "rb") as icon_file:
        return base64.b64encode(icon_file.read()).decode()

# Encode the audio and icon files to base64
audio_base64 = get_base64_audio(audio_file_path)
icon_base64 = get_base64_icon(icon_file_path)

# HTML and JavaScript to embed the audio player with autoplay and a fallback play button
audio_html = f"""
<audio id="background-audio" autoplay loop>
    <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
    Your browser does not support the audio element.
</audio>
<script>
document.addEventListener('DOMContentLoaded', function() {{
    var audio = document.getElementById('background-audio');
    var playPromise = audio.play();
    if (playPromise !== undefined) {{
        playPromise.then(_ => {{
            // Automatic playback started!
        }}).catch(error => {{
            // Auto-play was prevented
            console.log('Autoplay prevented by browser: ', error);
            // Show the play button
            document.getElementById('play-button').style.display = 'block';
        }});
    }}
}});
</script>
<img id="play-button" src="data:image/png;base64,{icon_base64}" style="display: none; cursor: pointer;" onclick="document.getElementById('background-audio').play()" alt="Play Music">
"""

# Embed the audio player in the Streamlit app
components.html(audio_html, height=150)

# Your Streamlit app content
st.write("This is an example Streamlit app with background music.")
