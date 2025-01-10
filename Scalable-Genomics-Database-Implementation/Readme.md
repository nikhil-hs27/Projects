#Make sure Python latest version is intalled
<ul>
   <li>Create .env file with contents
       MONGO_URL=mongodb://localhost:27017
       MONGO_DB_NAME=wheatsamples
    </li>
  
  <li>Run the command <strong>python -m venv .venv</strong></li>
  <li>Activate the virtual env by  running command(forwindows) <strong> .venv/Scripts/Activate.ps1</strong> </li>
  <li>Run the command <strong>pip install -r requirements.txt</strong> to install all libraries </li>
  <li>Installed Mongodb</li>
  <li>Run the command <strong>uvicorn app:app --reload</strong> to run the app</li>
</ul>
