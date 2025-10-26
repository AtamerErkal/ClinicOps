\# 🚀 KlinikOps MLOps Pipeline Project



\[!\[Pipeline Status](https://github.com/YOUR\_USERNAME/YOUR\_REPOSITORY/actions/workflows/mlops\_pipeline.yml/badge.svg)](https://github.com/YOUR\_USERNAME/YOUR\_REPOSITORY/actions/workflows/mlops\_pipeline.yml)

\[!\[Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python\&logoColor=white)](https://www.python.org/)

\[!\[DVC](https://img.shields.io/badge/Data%20Versioning-DVC-green?logo=dataversioncontrol\&logoColor=white)](https://dvc.org/)

\[!\[Azure](https://img.shields.io/badge/Cloud-Azure-0078D4?logo=microsoftazure\&logoColor=white)](https://azure.microsoft.com/)



This repository hosts an end-to-end MLOps (Machine Learning Operations) pipeline designed to process critical patient data, train a machine learning model, and archive the results as a deployable artifact.



The pipeline successfully combines code quality checks (CI), data version control (DVC), and model training (CD) into a single, unified workflow (`mlops\_pipeline.yml`).



\## 🎯 Project Goals and MLOps Cycle Overview



The core objective of this project is to ensure that the model and data are continuously updateable, traceable, and monitorable.



| Stage | Tools Responsible | Status | Description |

| :--- | :--- | :--- | :--- |

| \*\*1. CI/Code Quality\*\* | GitHub Actions, Flake8, Pytest | ✅ Complete | Checks new code for PEP8 compliance and executes all unit tests. |

| \*\*2. Data Versioning\*\* | \*\*DVC\*\*, \*\*Azure Blob Storage\*\* | ✅ Complete | Raw data (`Patient\_Stay\_Data.csv`) is untracked by Git and versioned securely on Azure Blob Storage. |

| \*\*3. Data Pull \& Prep\*\* | GitHub Actions, DVC Pull, `scripts/data\_processing.py` | ✅ Complete | The pipeline pulls versioned data from Azure, preprocesses it, and creates training/testing data splits. |

| \*\*4. Model Training\*\* | `scripts/train.py`, MLflow | ✅ Complete | Trains the model, logs parameters and metrics, and saves the model object to MLflow. |

| \*\*5. Model Archiving\*\* | GitHub Actions Artifacts | ✅ Complete | The resulting MLflow artifacts (`mlruns/`) are safely archived on GitHub for deployment readiness. |



---



\## 🛠️ Technology Stack



| Category | Tool | Badge |

| :--- | :--- | :--- |

| \*\*Version Control\*\* | Git | \[!\[Git](https://img.shields.io/badge/Control-Git-F05032?logo=git\&logoColor=white)](https://git-scm.com/) |

| \*\*Data Versioning\*\* | DVC | \[!\[DVC](https://img.shields.io/badge/Data%20Version-DVC-green?logo=dataversioncontrol\&logoColor=white)](https://dvc.org/) |

| \*\*Cloud Storage\*\* | Azure Blob Storage | \[!\[Azure Blob](https://img.shields.io/badge/Storage-Blob%20Storage-0078D4?logo=microsoftazure\&logoColor=white)](https://azure.microsoft.com/en-us/services/storage/blobs/) |

| \*\*CI/CD/MLOps\*\* | GitHub Actions | \[!\[GitHub Actions](https://img.shields.io/badge/Automation-Actions-2088FF?logo=githubactions\&logoColor=white)](https://docs.github.com/en/actions) |

| \*\*Model Tracking\*\* | MLflow | \[!\[MLflow](https://img.shields.io/badge/Tracking-MLflow-009988?logo=mlflow\&logoColor=white)](https://mlflow.org/) |

| \*\*Quality Control\*\* | Pytest / Flake8 | \[!\[Pytest](https://img.shields.io/badge/Testing-Pytest-0A9EDC?logo=pytest\&logoColor=white)](https://docs.pytest.org/en/7.1.x/) |



---



\## ⚙️ Local Setup and Execution



Follow these steps to set up and run the project in your local development environment.



\### 1. Prerequisites



\* Python 3.11

\* Git

\* DVC

\* An Azure account with a Storage Account and Access Key.



\### 2. Environment Setup



Create a virtual environment and install the necessary dependencies, including the Azure DVC plugin:



```bash

\# Activate your virtual environment (e.g., Conda)

conda activate klinikops\_env



\# Install all dependencies including the DVC Azure extension

pip install -r requirements.txt

pip install 'dvc\[azure]'

### 3. Initialize DVC and Configure Azure Remote



After installing dependencies, you must initialize DVC within your repository and link it to your Azure Blob Storage container (`dvc-remote`) to store and version your large data files.



\#### A. Initialize DVC and Define the Remote URL



First, initialize DVC and define the Azure remote URL using the `azure://` protocol. This format includes the Storage Account name (`clinicopsdvcstorage2025`) and the container name (`dvc-remote`).



```bash

\# 1. Initialize DVC (creates the .dvc/ directory)

dvc init



\# 2. Define the Azure Remote URL (uses the correct Azure URI format)

dvc remote add -d azure\_remote azure://clinicopsdvcstorage2025/dvc-remote

### 3. Configure Azure Authentication (Security Step)



To push and pull data, DVC needs your \*\*Azure Access Key\*\*. For security, we store this sensitive secret in the \*\*`.dvc/config.local`\*\* file, which is automatically ignored by Git.



1\.  \*\*Locate your Azure Storage Access Key\*\* (from the Azure Portal, under Access Keys for your storage account).

2\.  \*\*Manually edit\*\* the hidden local configuration file, \*\*`.dvc/config.local`\*\*, in your project's root directory.

3\.  Add the following content (replacing the placeholder with your actual key):



```ini

\# .dvc/config.local (MUST BE IGNORED BY GIT)

\['remote "azure\_remote"']

&nbsp;   account\_name = clinicopsdvcstorage2025

&nbsp;   account\_key = \[YOUR\_ACTUAL\_AZURE\_ACCESS\_KEY\_HERE]```

