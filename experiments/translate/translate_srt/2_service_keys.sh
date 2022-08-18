gcloud iam service-accounts keys create private_key.json \
--iam-account=translate-service-acct@translate-srt-1.iam.gserviceaccount.com
cp private_key.json ../private_key.json 
rm private_key.json