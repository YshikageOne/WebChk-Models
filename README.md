# WebChk-Models

> This repository contains model implementations used in **WebChk**, an AI-driven vulnerability scanner for web applications. It includes three model types for detecting security issues in web application code or payloads.

## Models Overview

### 1. BERT-based Vulnerability Detection

A transformer-based approach leveraging BERT to analyze code snippets or input data for potential vulnerabilities.

**Databases:**

- [DiverseVul Repository](https://github.com/wagner-group/diversevul) – Collection of diverse vulnerability examples.
- [PrimeVul Repository](https://github.com/DLVulDet/PrimeVul?tab=readme-ov-file) – PrimeVul dataset containing labeled vulnerability instances.

### 2. FP (Payload Fingerprinting) Model

A fingerprinting method that matches known payload patterns against inputs to flag malicious activity.

**Databases:**

- [PayloadsAllTheThings](https://github.com/swisskyrepo/PayloadsAllTheThings) – A comprehensive list of payloads for testing web applications.
- [Web Application Payloads Dataset (Kaggle)](https://www.kaggle.com/datasets/cyberprince/web-application-payloads-dataset) – Structured dataset of web attack payloads.

### 3. TCN (Temporal Convolutional Network)

A sequence modeling technique using temporal convolutions to detect anomalies over sequences of inputs or events.

**Database:**

- [Harvard Dataverse: Web Application Sequences](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/88M0DI) – Time-series dataset for web traffic and attack sequences.
