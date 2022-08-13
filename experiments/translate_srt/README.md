# GCLOUD WORKFLOW

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

# RUN SHELLS
- to create service acct: 
    . 1_create_service.sh
- to enable translate-service-acct: 
    . 1_enable_service.sh 
. 2_service_keys.sh # create service account keys
. 3_set_env.sh # Add env vars, pip install
. 4_disable_service # Disable service account

# FOR DEBUGGING
- Convert to UTF-16 LE BOM in Notepad++ if encoding not working



