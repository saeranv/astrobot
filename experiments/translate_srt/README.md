# GCLOUD WORKFLOW

# FOR TRANSLATION ASSUMING SERVICE ACCT/OLD PROJ EXISTS
gcloud init  # select config, project
optional check: gcloud projects list # ensure PROJ_ID=translate-srt-1
optional check: gcloud iam service-accounts list # ensure SA_NAME=translate-service-acct
. 1_enable_service.sh
- python -m translate_srt.app $PROJ_ID
. 4_disable_service # Disable service account


# REF
https://cloud.google.com/translate/docs/advanced/translate-text-advance

# INIT
gcloud init  # set configuration, select project.

# CHECK AND LINK BILLING ACCT
gcloud projects list # list all project ids
gcloud beta billing projects describe PROJECT_ID  # check if linked to billing
- If billingEnabled is false, go to dashboard, select Billing and link project.

# ENABLE API
- Go the dashboard and enable cloud-translate api

# CHECK SERVICE ACCOUNT
gcloud iam service-accounts list  # lists service accounts

# PRE-SHELL ASSUMPTIONS
- before run .sh ensure SA_NAME, PROJECT_ID
- default: SA_NAME=translate-service-acct
- default: PROJECT_ID=translate-srt-1

# SHELLS FOR CREATION
- to create service acct:
    . 1_create_service.sh
- to enable translate-service-acct:
    . 1_enable_service.sh
- to create service account keys
    . 2_service_keys.sh
- to add env vars, pip install
    . 3_set_env.sh # Add env vars, pip install

# FOR DEBUGGING
- Convert to UTF-16 LE BOM in Notepad++ if encoding not working



