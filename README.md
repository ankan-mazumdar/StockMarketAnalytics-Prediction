# StockMarketAnalytics-Prediction: A Comprehensive Data Engineering and Analysis Project

In this project, you will execute an End-To-End Data Engineering Project on Real-Time Stock Market Data using Kafka.

complete demo video- https://youtu.be/K2oqsRCKh5w?si=3lybRtGOHGj4CH2g

Try out the Deployed App

https://stockmarketanalytics-prediction.streamlit.app/#stocksavvy-services

## Architecture 
![image](https://github.com/user-attachments/assets/ff658beb-32d3-48f2-a8b9-870e52277213)


## Introduction

This project combines real-time stock market data processing with advanced analytics and prediction models. It leverages a robust tech stack including Apache Kafka, AWS services, and machine learning algorithms to provide valuable insights into stock market trends and predictions.

## Architecture Overview

The data pipeline follows these key steps:

1. Data Ingestion: Regional and global datasets are ingested into the system.
2. Kafka Producer: Data is sent to Apache Kafka for real-time processing.
3. Zookeeper: Manages Kafka clusters, ensuring high availability and load balancing.
4. Kafka Consumer: Consumes data from Kafka topics.
5. AWS S3: Stores processed data in JSON format.
6. AWS Glue Crawler: Automatically discovers and catalogs data schema.
7. AWS Glue Data Catalog: Maintains metadata information about the data.
8. Amazon Athena: Enables SQL queries on the data stored in S3.
9. Streamlit UI: Provides an interactive interface for data visualization and analysis.

## Key Features
- Dashboard and Insights: Access combined stock watchlists, historical data, and key insights.
- News and Sentiment Analysis: Get the latest news and sentiment analysis for your favorite stocks.
- Prediction Model: Predict next morning's opening prices using recent stock data.
- StockSaavy - Your Virtual Assistant: 24/7 assistant for stock market queries, insights, and analysis.

## Technology Stack

- Apache Kafka
- Amazon Web Services (AWS)
  - EC2
  - S3
  - Glue
  - Athena
- Python
- Streamlit
- Machine Learning Libraries (TensorFlow, Keras)

## Implementation Details

### Kafka Setup
- Kafka is installed and configured on an EC2 instance
- Zookeeper is started first, followed by the Kafka server
- Topics are created for data streaming

### AWS Configuration
- S3 buckets are set up to store the streaming data
- Glue Crawler is configured to scan S3 data and create schema
- Athena is set up for SQL-based data querying

### Streamlit UI
The user-friendly interface provides access to:
- Stock price predictions
- Interactive dashboards
- News and sentiment analysis
- AI-powered stock analysis assistant

## Project Workflow

1. Data is ingested and processed through Kafka
2. Processed data is stored in AWS S3
3. AWS Glue catalogs the data, making it queryable via Athena
4. The Streamlit UI allows users to:
   - View real-time and historical stock data
   - Access predictive models (LSTM, ARIMA)
   - Analyze market news and sentiment
   - Utilize the StockSavvy AI assistant for custom analysis

## Steps and screenshots

- Apache Kafka and Zookeeper's Role in Kafka

![image](https://github.com/user-attachments/assets/58b8df70-580e-4c25-ae02-860f59cc04a4)

EC2 VM Creation-

![image](https://github.com/user-attachments/assets/a06d9c63-6b89-4463-962a-1917bbaa29eb)

local terminal SSH connection-

![image](https://github.com/user-attachments/assets/ec8a9c75-abbf-4616-b46e-6a3c09571583)

Install Kafka using follwing commands 
wget https://downloads.apache.org/kafka/3.3.1/kafka_2.12-3.3.1.tgz
tar -xvf kafka_2.12-3.3.1.tgz

Zookeeper's Role in Kafka
Cluster Management: Zookeeper keeps track of Kafka brokers and manages metadata.
Leader Election: Ensures one broker acts as the leader for a specific partition.
Configuration Storage: Holds Kafka configuration data.

Start Zoo-keeper:
-------------------------------
Open another window to start kafka
cd kafka_2.12-3.3.1
bin/zookeeper-server-start.sh config/zookeeper.properties

But first ssh to to your ec2 machine as done above

Start Kafka-server:
----------------------------------------
Duplicate the session & enter in a new console --
export KAFKA_HEAP_OPTS="-Xmx256M -Xms128M"
cd kafka_2.12-3.3.1

![image](https://github.com/user-attachments/assets/2a3aecde-bb72-40b9-95fb-0ec3314e76ac)

![image](https://github.com/user-attachments/assets/d19ff60e-b4b2-429a-8fc6-ce63b69d525f)


![image](https://github.com/user-attachments/assets/7978c0ff-8a97-44ed-b573-179e3882ac9a)

-Initial blank S3 bucket and now SQL records in AWS Athena 

![image](https://github.com/user-attachments/assets/f3cde5c1-8d75-446a-b867-16010f46b0a6)




![image](https://github.com/user-attachments/assets/188bb2d6-34ac-4e43-83c9-1533752a93b9)

Note- It is pointing to private server , change server.properties so that it can run in public IP 

To do this , you can follow any of the 2 approaches shared belwo --
Do a "sudo nano config/server.properties" - change ADVERTISED_LISTENERS to public ip of the EC2 instance

Create the topic:
-----------------------------
Duplicate the session & enter in a new console --
cd kafka_2.12-3.3.1
bin/kafka-topics.sh --create --topic demo_testing2 --bootstrap-server {Put the Public IP of your EC2 Instance:9092} --replication-factor 1 --partitions 1

Start Producer:
--------------------------
bin/kafka-console-producer.sh --topic demo_testing2 --bootstrap-server {Put the Public IP of your EC2 Instance:9092} 

![image](https://github.com/user-attachments/assets/87435fdc-62c7-4a34-9884-21cbbd0f4c0b)

Start Consumer:
-------------------------
Duplicate the session & enter in a new console --
cd kafka_2.12-3.3.1
bin/kafka-console-consumer.sh --topic demo_testing2 --bootstrap-server {Put the Public IP of your EC2 Instance:9092}

![image](https://github.com/user-attachments/assets/71fbd585-9102-4d8e-abc9-ad011cfa68c7)

![image](https://github.com/user-attachments/assets/87435fdc-62c7-4a34-9884-21cbbd0f4c0b)

![image](https://github.com/user-attachments/assets/2ba41dda-ebc5-4a11-827f-a86a7de2a3c9)

![image](https://github.com/user-attachments/assets/b1e8225b-b13c-4bf5-a5f5-645b5a6ad920)


![image](https://github.com/user-attachments/assets/63ecff2e-bf24-4009-aa86-f34ada95cc96)

![image](https://github.com/user-attachments/assets/3e93d3cc-21c3-4104-aeb7-f87e8aea4738)

![image](https://github.com/user-attachments/assets/953d570b-0c9a-45b5-ae36-62dda02385d9)

![image](https://github.com/user-attachments/assets/9a2dfa41-198c-4667-82ed-ba8bbe91c27d)

S3 bucket gets loaded after running Producer and consumer -

![image](https://github.com/user-attachments/assets/405bace8-4516-492d-b96a-d76e340b4088)

![image](https://github.com/user-attachments/assets/0f537897-2523-467b-a7da-faaceb1826bf)

![image](https://github.com/user-attachments/assets/b0e5d27d-5708-4d2d-9a33-7d3ae3fded91)


![image](https://github.com/user-attachments/assets/ed92233a-21b8-415d-a545-3eb10d99fc52)

We will use AWS Crawler which will crawl thru entire file to fetch the schema, so that we can query using Athena on top of it.

![image](https://github.com/user-attachments/assets/9beb7b9f-ee6e-4489-ad4c-f2333b0c4b74)

Setup the account and download the kafka’s Access key ID & Secret access key And enter command aws configure


![image](https://github.com/user-attachments/assets/f04e82a5-ab1e-49f5-ab0e-ea065b247d84)

![image](https://github.com/user-attachments/assets/cf106c8c-58c8-4de0-9049-cab1ab61f3b9)

![image](https://github.com/user-attachments/assets/f049d81a-8e8d-4923-9918-2fc33643baed)

![image](https://github.com/user-attachments/assets/3568e4bd-9abb-4a2a-a6ea-0d6d05465797)

![image](https://github.com/user-attachments/assets/7745c5fe-1074-4cba-b311-0e9c5d669213)

![image](https://github.com/user-attachments/assets/c3fa4077-4fd8-470a-ae71-d3569a2ef2e8)

![image](https://github.com/user-attachments/assets/87eeb0fd-e753-4074-8bc9-6c05c8aebc3d)



![image](https://github.com/user-attachments/assets/e404dfe1-e07a-4a08-bac5-93c4985d1945)

![image](https://github.com/user-attachments/assets/fba59561-0475-4bba-8d37-4d1706dd5090)

![image](https://github.com/user-attachments/assets/476e4a73-f9ec-4d7d-b72c-efee60c8a05e)



![image](https://github.com/user-attachments/assets/df0776ea-3cef-4f90-92f9-1761d96314b5)

![image](https://github.com/user-attachments/assets/86298201-7a1c-4069-9acf-fb78e6f801df)

![image](https://github.com/user-attachments/assets/00a0cb77-6d66-48ad-9af5-2b7f6ff2af83)

![image](https://github.com/user-attachments/assets/39f05587-7acf-4d2d-b085-65ec6f157a02)

![image](https://github.com/user-attachments/assets/3c9e59e3-8900-4a22-b3f2-7129bb65bd03)

![image](https://github.com/user-attachments/assets/486af9d2-e943-4363-a706-731bd833067a)

![image](https://github.com/user-attachments/assets/beb5fba2-ff9c-4592-8c80-703240f070c6)

![image](https://github.com/user-attachments/assets/128399de-0f06-45eb-b13e-078176c3a80e)

![image](https://github.com/user-attachments/assets/5baac2c9-1936-4cb6-bbe5-ac804d4f9a21)

![image](https://github.com/user-attachments/assets/a718c7a3-6f3c-4817-82d8-b824dae1a119)

