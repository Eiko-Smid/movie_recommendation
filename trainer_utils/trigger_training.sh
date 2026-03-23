#!/bin/sh

# Mark training start in logs
echo "$(date '+%Y-%m-%d %H:%M:%S') - Trainer start" >> /var/log/cron.log

# Check if service token exists
if [ -z "$API_SERVICE_TOKEN" ]; then
  echo -e "API_SERVICE_TOKEN is not set! Exiting." >> /var/log/cron.log
  exit 1
else
  echo -e "API_SERVICE_TOKEN is set. Training starts..." >> /var/log/cron.log
  # Measure start time
  START_TIME=$(date +%s)

  START_TIME=$(date +%s)
  curl -sS -X POST http://api:8000/train/train_model \
    -H "api-service-key: $API_SERVICE_TOKEN" \
    -H "Content-Type: application/json" \
    -d @/app/trainer_utils/train_payload.json \
    >> /var/log/cron.log 2>&1

  # Compute training duration
  END_TIME=$(date +%s)
  DURATION=$((END_TIME - START_TIME))
  echo "$(date '+%Y-%m-%d %H:%M:%S') - Training completed in ${DURATION} seconds." >> /var/log/cron.log
  fi
