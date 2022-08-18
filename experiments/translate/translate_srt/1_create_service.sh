gcloud iam service-accounts create translate-service-acct \
--description="" \
--display-name=translate-service-acct

gcloud projects add-iam-policy-binding translate-srt-1 \
    --member="serviceAccount:translate-service-acct@translate-srt-1.iam.gserviceaccount.com" \
    --role="roles/cloudtranslate.user"
