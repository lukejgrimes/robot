<h1>Robot</h1>
<h3>Welcome to The Robot Repository</h3>
<p>Robot uses FaceNet and ANNOY (Approximate Nearest Neighbor) to identify human faces and remember people it has spoken to previously.</p>
<p>Robot uses ChatGPT to speak to users</p>

<h3>Future plans:</h3>
<ul>
    <li>Build small physical body for robot with autonomus movement capabilities</li>
    <li>Use collected conversation data to fine tune our speaking model and produce more natural speech</li>
</ul>

<h3>Installation:</h3>
<ol>
    <li>Create a virtual environment</li>
    <li>Git clone the repository</li>
    <li>pip install -r requirements.txt</li>
    <li>Create a .env file and add the following:<li>
    <ul>
        <li>OPENAI_API_KEY= Your API key from OpenAI</li>
        <li>ANNOY_INDEX= File path which you would like to save you annoy index file</li>
        <li>GOOGLE_CLOUD_CREDS= File path to you Google Cloud Credentials (Must sign up for google cloud and enable text-to-speech and speech-to-text APIs)</li>
    </ul>
</ol>