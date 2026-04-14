# AI WATER ANOMALY DETECTION SYSTEM
This project implements an **AI-powered anomaly detection system** for water infrastructure using the **Isolation Forest algorithm**. It analyzes time-series sensor data to detect abnormal patterns such as **water leaks** and **cyber-physical attacks**.

---
 
## Table of Contents
 
1. [The Problem](#the-problem)
2. [The Dataset](#the-dataset)
3. [How It Works](#how-it-works)
4. [Setup & Installation](#setup--installation)
5. [Running the Project](#running-the-project)
6. [Concepts Explained](#concepts-explained)
7. [Results](#results)
8. [What I Learned](#what-i-learned)

---

## The Problem

Municipalities in South Africa are estimated to lose between 47% to 56% of treated water because of leaks, pipe bursts, and illegal connections, this has lead to approximately R14 billion being wasted annually.
Most systems currently implemented are reactive so they are only able to identify when a leak happens only once it happens for example after a street floods, spike in water bills, complaints logged by community members.

The aim of this project is to provide a proactive solution by using AI (machine learning) to be able to identify the early signs of abnormal water to loss.

---
## The Dataset

This project makes use of the **BATADAL — Battle of the Attack Detection Algorithms** benchmark, which represents an water distribution network.
- Training Set (Dataset 03): 8,761 hours of purely normal operation. This is used to establish a mathematical baseline of the system's "pulse."
- Test Set (Dataset 04): A period containing unknown anomalies, including simulated pipe fractures and cyber-physical tampering.

---

## How It Works

The system follows a semi-supervised anomaly detection pipeline:

1. Learning Normalcy: The model "studies" the normal hydraulic patterns (flows, pressures, tank levels) from Dataset 03.
2. Feature Engineering: Raw sensor data is transformed into hydraulic indicators like Minimum Night Flow (MNF).
3. Anomaly Scoring: The custom Isolation Forest measures how "different" new data is from the learned baseline.
4. Thresholding: The system automatically flags suspicious events for human intervention.

---

## Setup & Installation

### Step 1: Clone the repository
```bash
git clone https://github.com/YourName/AI-Water-anomaly-detection-System.git
cd AI-Water-anomaly-detection-System
```
### Step 2: Install dependencies:
```bash
pip install numpy pandas matplotlib 
```
---

### Running the project

To train the model on the normal baseline and scan the test data for anomalies, run:

```bash
python main.py
```
After execution, a summary will be printed to the terminal, and all visual reports will be saved in the **outputs/** directory.

---

## Concepts explained

## 1.Supervised vs Unsupervised vs Semi-Supervised learning

In **supervised learning**, you give the algorithm labelled examples: "this is a leak, this is not a leak." The algorithm learns the difference.
 
In **unsupervised learning**, you don't have labels (or you don't trust them fully). The algorithm must find patterns on its own. Isolation Forest is unsupervised — it doesn't need to be told what a leak looks like. It just asks: "what's unusual?"

In **semi-supervised learning**, this is where models are trained s using a small amount of labeled data alongside a large volume of unlabeled data.

This is important for water networks because:
- Leaks are rare → very few labelled examples
- New types of anomalies may not match known patterns
- Real-time systems can't wait for an operator to label each reading
- 
---

## 2.What is an Isolation Forest

___The purpose Isolation Forest is: Is this data point easy or hard to isolate___

Isolation Forest isolates anomalies by randomly partitioning data. Because anomalies are "few and different," they are isolated much faster (shorter path length) than normal points.The anomaly score $s(x, n)$ is calculated as:$$s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$$Where $E(h(x))$ is the average path length and $c(n)$ is the normalization constant for a failed search in a Binary Search Tree.

```
NORMAL DATA POINT:                    ANOMALY:
─────────────────────────────         ─────────────────────────────
Surrounded by similar points          Far from other points
 
    ● ● ●                                 ●
   ●  ◉  ●    ← target point                         ◉  ← target
    ● ● ●
 
Split 1: picks a line                 Split 1: picks a line
  → still has neighbours                → already alone!
 
Split 2, 3, 4...                      Done in 2 splits!
  → many splits needed
 
Long path = NORMAL                    Short path = ANOMALY
```
 
The algorithm builds many random trees. Each tree randomly:
1. Picks a feature (e.g. flow rate)
2. Picks a random split value (e.g. 5.3 LPS)
3. Separates points: above vs below the split
4. Repeats until each point is alone
 
**Average path length across all trees = anomaly score**
- Long path (hard to isolate) = normal reading
- Short path (easy to isolate) = anomalous reading
 
---
## 3. Minimum Night Flow (MNF)

In the water industry, the "rising floor" is a classic leak signature. We calculate the minimum flow between 01:00 and 04:00 AM. If this "minimum" starts to drift upward over time, it indicates a background leak that never stops, even when the city is asleep.

---

## 4. RobustScaling vs StandardScaling
Standard scaling transforms data using the mean (μ) and standard deviation (𝜎):

z= x−μ / σ

The issue:
- In datasets containing anomalies (such as pipe bursts or water theft), these extreme values are included when calculating the mean and standard deviation.

The effect:
- Outliers shift the mean and increase the standard deviation. This spreads or “smears” the data, making anomalies appear less distinct and more like normal variations. This phenomenon is known as outlier leakage.
---
Robust scaling uses the median and interquartile range (IQR):

x scaled​ = x−median / IQR where IQR = Q3 - Q1

Why it works:
 - The median and IQR are resistant to extreme values. Even in the presence of significant anomalies, such as a large pipe burst, these statistics remain stable.

The result:
- The baseline reflects true normal behavior. As a result, anomalies stand out more clearly because they do not influence how the data is centered or scaled.

---

### Results:

## 1. Main Detection View
<img width="2383" height="1219" alt="plot1_detection" src="https://github.com/user-attachments/assets/ad21bca0-bca9-43f5-bbc9-c738aa7719e5" />
This plot identifies specific hours where flow patterns broke the learned "Normal" baseline.

The red markers indicate points where the Isolation Forest found a path length so short it had to be an anomaly. Clusters of red dots during high-flow periods suggest sustained leaks or tampering.

---
## 2. Anomaly Score Distribution
<img width="1485" height="735" alt="plot2_score_distribution" src="https://github.com/user-attachments/assets/39d2d7ed-e6af-42e8-afc7-96bd70017038" />

This histogram shows the model's ability to distinguish between classes.

Want to see a clear separation. The large blue hump is the "Normal" data; the "tail" on the left represents the anomalous events. A distinct tail indicates the model is confident in its detections.

---
## 3. Multi-Sensor Grid
<img width="2386" height="1330" alt="plot3_multi_sensor" src="https://github.com/user-attachments/assets/0420fe47-d261-407d-869a-c8389b9d2403" />

This allows us to identify the impact of a detected event across different pumps.

What is being observed is that if all pumps show red dots simultaneously, the issue is system-wide (e.g., a major burst or power outage). If only one pump shows red dots, the fault is localized to that specific branch.

---
## Monthly Anomaly Count
<img width="1335" height="735" alt="plot4_monthly_anomalies" src="https://github.com/user-attachments/assets/50bf2752-c30c-4ac6-8df3-34c07780515d" />

A strategic view used for municipal maintenance and budgeting.

Bars are color-coded by severity. Red bars indicate months with the highest infrastructure stress, allowing engineers to correlate leaks with seasonal factors like temperature shifts.

---
## 5. Minimum Night Flow Analysis
<img width="2083" height="585" alt="plot5_night_flow" src="https://github.com/user-attachments/assets/83ecd495-7424-4dbc-ac77-62f2f9c0b75b" />

The primary hydraulic indicator for background leaks.

How to read it: This plot focuses on the 01:00 AM - 04:00 AM window. If the "floor" of the water flow (the green area) drifts upward over weeks, it is a mathematical signature of a growing underground fracture.

---

### What I learnt

- Algorithmic Foundations: Implementing an Isolation Forest from scratch gave me a deep understanding of how recursive partitioning works and how the c(n) normalization constant is used to compute anomaly scores.
- Unsupervised Anomaly Detection: I learned how Isolation Forest detects anomalies without relying on labeled data, instead identifying unusual patterns based on how quickly data points are isolated.
- Domain Knowledge Integration: I realized that machine learning models become far more effective when combined with domain-specific insights, such as Minimum Night Flow (MNF), which is rooted in water engineering rather than generic ML techniques.
- Outlier-Resistant Scaling: I understood the importance of using Robust Scaling instead of Standard Scaling, as mean-based methods can dilute anomalies, while median-based scaling preserves them.
- System-Level Thinking: Working with the BATADAL dataset taught me to analyze the system as a whole, viewing the municipality as an interconnected hydraulic network rather than isolated sensors
  
​
