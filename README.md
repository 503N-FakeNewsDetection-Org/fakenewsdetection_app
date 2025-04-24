# Fake News Detection App

### Changes to allow for full testing:
1. Both retraining scripts' **accuracy** and **f1** thressholds are set to 0.1, originally they are much higher. This is for testing purposes so the retraining workflow always promotes the shadow model weights.
2. Upon performing the **first** retraining workflow for each model, the data on azure (data through user feedback, which I have populated right now so you can test it) will be **deleted**. In case you would like to test again, repopulate the containers by querying the actual application and choosing to give feedback. Make sure they are **enough** (at least 20 for the text model, at least 10 AI images and 10 human images for the image model) and that they are **UNIQUE** (Identical queries will not be saved!)
3. In case of anything, kindly contact me since only I have access to my azure account.

## Project Overview
This project is a Minimum Viable Product (MVP) designed to detect fake news using machine learning models. It includes both image and text models to analyze news content and determine its authenticity. The MVP status means that while the core functionality is present, there are areas for further development and optimization, such as security, data storage, and human intervention.

### Models
- **News Headlines Classifier**: Pre-trained BERT model, added classification head and performed training through freezing the BERT weights while only training the classification head. Trained on ~60,000 headlines (balanced split). It is worth noting that the service this model provides is SOLELY headlines classification. This means when us, humans, are unable to determine if the headline is real or fake at first glance, we would then ask the model for a reliable prediction.

From my testing, the model cannot understand numerical values too well. For example, querying "Trump enforces 5% taxes on China" would yield a very similar result compared to "Trump enforces 90000000% taxes on China." The model expresses superior capabaility at understand the English language, however.

- **Online Image Classifier**: Pre-trained SigLIP2 model, trained on ~120,000 images (balanced split) (no weights freezing, full model is trained). It is worth noting that this model's context of AI images are ones created by other, SoTA models. Officially: "The model demonstrates strong capability in detecting high-quality, state-of-the-art AI-generated images from models such as Midjourney v6.1, Flux 1.1 Pro, Stable Diffusion 3.5, GPT-4o, and other trending generation models."

From my testing, it is CRUCIAL that the entire image is shown to the model (no cropping).

### Key Features
- **Weekly Retraining**: The models are retrained weekly to ensure they remain up-to-date with the latest data. This process includes running unit and integration tests to verify the integrity and performance of the models. MLFlow was also integrated but not committed to the repo. 
---Author's note: I attempted to set the tracking uri of MLFlow onto azure, but it would not work due to version and dependency issues. Regardless, the MLFLow framework is there.

- **Shadow Deployment**: Retraining script pushes shadow model to azure blob storage, sends authenticated admin POST request to internal application (sets it to SHADOW mode)
The app will now forward queries to the shadow model, agreement rate with prod_model & failed req rate scraped by prometheus.

- **Weekly Evaluate (Day after retraining)**: The day's metrics are stored on prometheues, pulled by GitHub actions, runs tests to see if shadow was good. Overwrite old production model weights if good, delete shadow if bad.

- **Admin Routes**: Special admin routes are available to facilitate operations without the need to redockerize and redeploy to Azure. These routes allow for efficient management and updates. These routes are /admin/<text/image>/<role/reload/status>, which:
1. **/role**: Change the container's role (prod/shadow mode) and reload it. It downloads both shadow and production weights, and forwards all requests to the shadow model.
2. **/reload**: Reload the container (it downloads the current production weights again)
3. **/status**: Return the current role of the container.

- **Data Handling**: User data is saved if they are certain about the news authenticity. For MVP purposes, archives are stored on GitHub, but the system can be configured to store archives on Azure. Note that blob storage is not optimal for CSV data, and JSON username/password files are not ideal for security and scalability.


### How It Works
1. **Data Collection**: The system collects news headlines and images for analysis.
2. **Model Analysis**: The text and image models analyze the content to determine its authenticity.
3. **User Interaction**: Users can interact with the system to verify news authenticity and provide feedback.
4. **Admin Management**: Admins can manage the system through special routes, allowing for updates and maintenance without full redeployment.

This overview provides a high-level understanding of the project's capabilities and limitations as an MVP. Further sections will detail the setup and reproducibility of the system.

## Model Weights
The model weights are not included in the repository due to size constraints. You can download them from the following Google Drive link:
[Download Model Weights](https://drive.google.com/drive/folders/1Gua-mF2ZHvxtM6qFs273qauhS7QULVxs?usp=drive_link)
In case of localhost testing, place model.pt inside text_model/ and image.pt inside image_model/
If you plan to dockerize and push to azure, skip downloading the weights entirely.

## Setup and Reproducibility

### Prerequisites
- Python 3.x
- Docker (to containerize)
- Azure CLI (for deployment on Azure)

### Local Setup for localhost testing
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd fakenewsdetection_app
   ```

2. **Install Dependencies**:
   Install the required Python packages using:
   ```bash
   pip install -r requirements.txt
   ```
   For each endpoint directory (main/,text_model/,image_model/)

3. **Environment Variables**:
   For MVP purposes, the Azure storage connection string and admin token are hardcoded in the scripts. In an enterprise deployment, these would be stored in GitHub secrets and accessed via GitHub Actions. (or via another mechanism)

4. **Run the Service Locally**:
   Image/Text: Uncomment the _load_local() functions in both image and text app.py scripts, uncomment the LOCAL_WEIGHTS_DIR variable up top, replace "prod_sd  = download_blob(PROD_BLOB)" with "prod_sd  = _load_local(PROD_BLOB) or download_blob(PROD_BLOB)", add a uvicorn.run command at the bottom to execute the apps on localhost.

   Main: Set <TEXT/IMAGE>_SERVICE_BASE = http://localhost:<port>, add a uvicorn.run command at the bottom to execute the main gui/proxy on localhost.

   Start the application using:

   ```bash
   python text_model/app.py 
   python image_model/app.py 
   python main/app.py
   ```
   OR: Dockerize and run locally (from repo root):
   ```bash
   docker-compose -f main/docker-compose.yml up --build
   ```

### GitHub Actions Workflows: (CI/CD)
The service is LIVE right now at https://gateway.wonderfulplant-23b2d8b0.francecentral.azurecontainerapps.io/
Prometheus is LIVE right now at https://prom.wonderfulplant-23b2d8b0.francecentral.azurecontainerapps.io/
Dr. Ammar will also be added to the GitHub repo collaborators.
Optionally, setup the entire ACR and ACA frameworks manually by containerizing the microservices, push them to ACR, create ACA applications to host it and view logs to check status updates (reloads, container role updates, etc..) (Note that this would require the correct containers and model blob names to be in your azure blob storage)

Practically, use the already available application on azure.

1. **Text Dataset**: The original training data (original_fake.csv and original_real.csv) are available in the repo. (I know we should not store data on GitHub for obvious reasons, but seeing as we are in a controlled project environment, I hope it would be okay). Sample user data is currently stored on azure blob storage (and you are free to add your own inputs through the application process). On retrain they are pulled, concatenated, shuffled, and ready.

2. **Image Dataset**: For MVP purposes, only 10 images are currently available in each of datasets/image_data/ai_original and datasets/image_data/hum_original. Sample user data is currently stored on azure blob storage (also feel free to add your own through the application process). On retrain they are correctly pulled, labelled, shuffled, and ready.

3. **GitHub Workflows**:

   | Category | Workflow File | What it Does | Schedule | Manual Trigger |
   |----------|---------------|--------------|----------|----------------|
   | **Retrain** | `.github/workflows/text_retrain.yml` <br> `.github/workflows/image_retrain.yml` |  *Retrain the text or image model from scratch each week.*  <br> • Checks out code, installs dependencies  <br> • Downloads current production weights + user‑contributed data + original training data (from GitHub, can easily be configured to download also from azure) from Azure Blob Storage  <br> • Runs **unit** and **integration** tests to guarantee a clean baseline  <br> • Launches the Python `retraining/*.py` script which produces a **shadow model**  <br> • Archives the user CSV / images back into the repo for provenance  <br> • Sends an authenticated POST to the running service (`/admin/<svc>/role`) to flip the container into **shadow** mode so new traffic is scored by the freshly trained model. | `cron: 0 0 * * 0` (Sunday 00:00 UTC for text model) <br> `cron: 30 0 * * 0` (Sunday 00:30 UTC for image model) | `workflow_dispatch` |
   | **Evaluate** | `.github/workflows/text_evaluate.yml` <br> `.github/workflows/image_evaluate.yml` |  *Decide whether the shadow model becomes production.*  <br> • Runs the morning after retraining  <br> • Pulls **Prometheus** via `$PROM_URL` to fetch:  <br>   `model_inferences_total{role="prod"}`  <br>   `shadow_inferences_total`  <br>   `shadow_agree_total`  <br> • Ensures shadow saw ~the same request volume and ≥ 85 % agreement with production  <br> • If criteria met ⇒ uploads **shadow blob → production blob** in Azure Storage  <br> • Else ⇒ deletes the shadow blob  <br> • Finally toggles the service back to **prod** mode via the admin route. | `cron: 0 0 * * 1` (Monday 00:00 UTC for text) <br> `cron: 30 0 * * 1` (Monday 00:30 UTC for image) | `workflow_dispatch` |
   | **End‑to‑End (E2E)** | `.github/workflows/e2e.yml` |  *Continuous availability check.*  <br> • Every hour it calls the gateway's admin endpoints for both services  <br> • Verifies HTTP 200; fails if any non‑success status is returned  <br> • Acts as a lightweight uptime monitor and guard‑rail. | `cron: 0 * * * *` (hourly) | `workflow_dispatch` |

   All workflows can be started manually from the **Actions** tab in GitHub via the "Run workflow" dropdown (enabled by the `workflow_dispatch` trigger). This allows ad‑hoc retraining, evaluation, or health checks without waiting for the next scheduled window.
   **NOTE THAT** we assume the base model is the ground truth, more reliably, we introduce a human-in-the-loop to monitor the data flow and manually test the shadow model on specific tests. This approach yields much higher confidence to promote/discard the shadow model.

   ###Testing the CI/CD:
   1. Just manually trigger the train workflow, send a query to the live apps, go over to the live prometheus URL (note that metrics on prometheus do not reset, by default they have a 15 day retention period), then query the metrics to verify the shadow model is up and running (shadow metrics begin with "shadow", production metrics begin with "model"). The workflow sends an authenticated POST request to the container, letting it know it is now in "shadow" mode.
   2. After that, run the evaluate script and watch it make the decision to promote or discard the shadow model. The workflow then sends a POST request to the container letting it know it is back in production mode, and it downloads the current production weights (shadow weights will replace production weights in case of promotion).
   3. Optionally, you are free to run the e2e.yml script at any time to check out the container's current mode. (Sends a GET request to both containers to query their STATUS).

   **NOTE THAT** as stated previously, the image model's original training data is not available for MVP purposes, while the text model's original training data is available in datasets/text_data/
   Do not forget the notes at the beginning of this README concerning data on azure!

For any questions, I am available by my work email - mwj06@mail.aub.edu