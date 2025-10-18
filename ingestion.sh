#!/bin/bash

# ------------------------------------------------------------
# NYC Taxi Dataset - Data Ingestion Script (with Logging)
# ------------------------------------------------------------
# This script uploads dataset files from the local filesystem
# into the Hadoop Distributed File System (HDFS).
# It verifies that the data is successfully stored by listing
# and displaying HDFS directory contents.
# ------------------------------------------------------------

# Exit immediately if a command fails
set -e

# ====== CONFIGURATION ======
LOCAL_DIR="/data"               # directory inside container containing files
HDFS_DIR="/taxi_data"           # target directory in HDFS
LOG_FILE="/scripts/ingestion_log.txt"

# ====== START LOGGING ======
echo "----------------------------------------------" | tee -a $LOG_FILE
echo "🚀 Starting Data Ingestion: $(date)" | tee -a $LOG_FILE
echo "Local Directory: $LOCAL_DIR" | tee -a $LOG_FILE
echo "Target HDFS Directory: $HDFS_DIR" | tee -a $LOG_FILE
echo "----------------------------------------------" | tee -a $LOG_FILE

# Step 1: Create HDFS directory
echo "📁 Creating HDFS directory (if not exists)..." | tee -a $LOG_FILE
hdfs dfs -mkdir -p $HDFS_DIR 2>&1 | tee -a $LOG_FILE

# Step 2: Upload files to HDFS
echo "⬆️  Uploading files from $LOCAL_DIR to $HDFS_DIR..." | tee -a $LOG_FILE
hdfs dfs -put -f ${LOCAL_DIR}/* $HDFS_DIR/ 2>&1 | tee -a $LOG_FILE

# Step 3: Verify upload
echo "✅ Verifying files in HDFS directory..." | tee -a $LOG_FILE
hdfs dfs -ls $HDFS_DIR 2>&1 | tee -a $LOG_FILE

# Step 4: Show HDFS directory usage
echo "📊 HDFS directory usage:" | tee -a $LOG_FILE
hdfs dfs -du -h $HDFS_DIR 2>&1 | tee -a $LOG_FILE

# Step 5: Completion message
echo "🎉 Data ingestion completed successfully at $(date)" | tee -a $LOG_FILE
echo "Logs saved to: $LOG_FILE"
echo "----------------------------------------------" | tee -a $LOG_FILE
