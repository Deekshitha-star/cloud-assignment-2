Github Link: https://github.com/Deekshitha-star/cloud-assignment-2
Docker Image Link: https://hub.docker.com/r/deekshitha523/cloud-assignment-2

## Cloud Assignment 2 - Instructions to run

### Step 1: Acquire AWS Credentials
1. Log into your AWS Educate account and start the lab.
2. Click on the "AWS Details" icon.
3. Download the .pem file for SSH access to EC2 instances.
4. Note down the Access key and Secret key (to be used later).

### Step 2: Upload Datasets to S3
1. Upload `TrainingDataset.csv` and `ValidationDataset.csv` into the a s3 at the location: s3://dm752bucket/cloud-assignment-2/.

### Step 3: Launch an EMR Cluster
1. Access the AWS console via the "AWS" link at the top right of your AWS Educate account.
2. Select "EMR" from the services menu and choose "create cluster".
3. Ensure Spark is included in applications. Set the core instances size to 4, disable auto-termination, and select your SSH key.
4. Create the cluster.
5. After cluster creation, open the master instance's security group and add SSH permissions for your IP address.

### Step 4: Prepare PEM File
1. On your local machine, open a terminal or command prompt.
2. Move the downloaded .pem file to a secure location and set its permissions:
   ```bash
   chmod 400 /path/to/your_key.pem
   ```

### Step 5: Transfer Model Training Jar
1. In a terminal or command prompt, use `scp` to copy the "cloud-assignment-2-0.0.1.jar" jar to the master node:
   ```bash
   scp -i /path/to/your_key.pem /path/to/file/cloud-assignment-2-0.0.1.jar hadoop@master_instance_public_dns:/home/hadoop
   ```
   Replace placeholders with actual values.

### Step 6: SSH into Master Instance
1. Use SSH to connect to the EC2 master instance:
   ```bash
   ssh -i /path/to/your_key.pem hadoop@master_instance_public_dns
   ```

### Step 7: Run Training Code
run the below command for training:
```bash
spark-submit --class com.example.Training --master yarn cloud-assignment-2-0.0.1.jar
```
- Spark will train the model using the core nodes and save the best model to `s3://dm752bucket/cloud-assignment-2/saved_model`.

### Step 8: Run Prediction Code
run the below command for testing:
```bash
spark-submit --class com.example.Testing --master yarn cloud-assignment-2-0.0.1.jar s3://dm752bucket/cloud-assignment-2/saved_model s3://dm752bucket/cloud-assignment-2/ValidationDataset.csv 
```

### Step 9: Docker for Prediction
1. Log into Docker:
   ```bash
   docker login
   ```
2. Navigate to the repository's directory on your machine.
3. Build the Docker image:
   ```bash
   docker build -t deekshitha523/cloud-assignment-2 .
   ```
4. Push the image to Docker Hub:
   ```bash
   docker push deekshitha523/cloud-assignment-2
   ```

### Step 10: Perform Prediction using docker
1. Create an EC2 instance.
2. Download the best model from S3 to your local machine.
3. Upload the model and the `ValidationDataset.csv` file to the EC2 instance.
4. SSH into the EC2 instance:
   ```bash
   ssh -i /path/to/your_key.pem ec2_user@master_instance_public_dns
   ```
5. Pull the Docker image for prediction:
   ```bash
   docker pull deekshitha523/cloud-assignment-2
   ```
6. Run the prediction using the Docker container:
   ```bash
   docker run -v $(pwd):/data deekshitha523/cloud-assignment-2  /data/saved_model  /data/ValidationDataset.csv
   ```


