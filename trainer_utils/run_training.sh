#!/bin/sh

# --------------------------------------------------
# General:
# Sets up cron job AND runs cron daemon
# --------------------------------------------------

# Write cron job
echo "0 2 * * * /app/trainer_utils/trigger_training.sh" > /etc/crontabs/root

echo "Cron job scheduled (02:00 UTC)"

# Start cron in foreground
crond -f -l 2