# **Mining Customer Sentiment in Amazon Reviews Using LDA and BERTopic**

This project analyzes customer sentiment and hidden thematic structure in Amazon reviews for Nike products, spanning footwear, apparel, accessories, and related product categories. Using a fully reproducible natural language processing (NLP) pipeline, the project applies both **classical topic modeling (LDA)** and **modern embedding-based modeling (BERTopic)** to uncover product themes, performance attributes, aesthetic trends, and consumer pain points. It also integrates sales rank, rating distributions, and category-level exploratory analysis to tie textual themes to marketplace behavior.

---

## **Project Goals**

* Extract meaningful themes from raw Amazon reviews
* Compare LDA vs BERTopic for topic quality and interpretability
* Analyze sentiment, ratings, and sales-rank distributions
* Identify product clusters such as performance footwear, streetwear, outdoor gear, authenticity concerns, and niche sizing issues
* Provide actionable insights relevant to consumer behavior and brand perception

This project can serve as a template for applied text mining, e-commerce analytics, and customer-experience research.

---

# **Project Structure**

```
.
├── notebooks/
│   └── Mining_Customer_Sentiment.ipynb
├── data/
│   └── (Amazon review JSON files or cleaned datasets)
├── models/
│   ├── lda_model/
│   └── bertopic_model/
├── visuals/
│   ├── intertopic_maps/
│   ├── rating_plots/
│   └── sales_rank_plots/
└── README.md
```

---

# **Methods & Pipeline**

## **1. Data Loading & Cleaning**

Raw review text is extracted from Amazon review JSON files.
Key metadata includes:

* reviewText
* product category
* rating
* salesRank

Cleaning steps (spaCy-based):

* Tokenization
* Lemmatization
* Removing punctuation, numbers, and short tokens
* Lowercasing and whitespace normalization

---

## **2. Text Preprocessing (spaCy + gensim)**

```python
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
```

Custom preprocessing ensures:

* normalization of word forms
* uniform tokenization
* compatibility with gensim dictionary and bag-of-words structures

---

## **3. LDA Topic Modeling**

We build a **50-topic** LDA model using gensim:

* `alpha = 1/num_topics`
* `eta = 0.01`
* 500 iterations
* 10 passes

Outputs include:

* topic-term matrix
* document-topic distributions
* **pyLDAVis intertopic distance map**
* topic summaries and cluster interpretations

---

## **4. BERTopic Modeling**

BERTopic is used to:

* generate transformer-based embeddings
* cluster documents using HDBSCAN
* construct class-based TF-IDF topic terms
* produce interactive HTML visualizations
* extract hierarchical topic relationships

BERTopic excels at capturing:

* context
* multilingual patterns (e.g., Spanish segments)
* subtle product-level themes

---

## **5. Topic Comparison: LDA vs BERTopic**

Both models were compared for:

* coherence
* cluster separation
* linguistic clarity
* stability
* interpretability of product themes
* cross-category alignment

Key Findings:

* **LDA** captures clean, interpretable “bag-of-words” topics
* **BERTopic** captures deeper semantic relations and mixed-language patterns
* Both reproduce similar high-level structure despite differences in topic indexing

---

# **Topic Cluster Interpretation**

Topic clusters align with:

* **Performance footwear** (running, basketball, cleats, pronation control)
* **General lifestyle footwear and streetwear** (kicks, colors, aesthetics)
* **Outdoor/winter products** (boots, socks, waterproofing, warmth)
* **Authenticity and fulfillment concerns** (fake products, shipping issues)
* **Activity-specific gear** (tennis grip, basketball dunks, clay courts)
* **Size/fit-related niche issues** (wide/narrow sizing, children’s footwear)
* **Bags and functional accessories** (wallets, backpacks)
* **Positive sentiment clusters** (“love,” “dope,” excitement-driven reviews)
* **Spanish-language review segments**

Despite varying random seeds, topic groups remain **structurally self-similar** across runs.

---

# **Exploratory Data Analysis**

## **1. Rating Distributions**

Ratings are heavily skewed positive across categories (Shoes, Clothing, Watches).
Most reviews cluster at 4–5 stars.

## **2. Sales Rank Distributions**

Sales ranks are strongly right-skewed:

* Footwear and Clothing: large volume but long tails
* Watches: more stable mid-tier ranks
* Health & Personal Care: low penetration and high volatility

## **3. Correlation Between Rating & Sales Rank**

Correlation ≈ **0** for most categories.
Conclusion:
**Ratings do not predict sales performance**.
Brand visibility and category competitiveness matter more.

---

# **Visuals Included**

* LDA intertopic distance map (HTML)
* BERTopic topic visualization (HTML)
* Sales rank histograms per category
* Rating distributions
* Ridgeline plots (log-scale sales rank)
* Violin plots (sales rank by rating)

---

# **Key Insights**

* Nike review themes cluster into consistent product categories (shoes, apparel, accessories)
* Both LDA and BERTopic reveal shared macrostructure
* BERTopic adds rich multilingual and contextual detection
* Ratings are overwhelmingly positive but not meaningful predictors of sales
* Sales rank distributions show clear differences by product type
* Authenticity and fulfillment concerns stand out as a persistent review theme

---

# **How to Run**

### **Install Dependencies**

```bash
pip install gensim spacy bertopic plotly pyldavis
python -m spacy download en_core_web_sm
```

### **Run the Notebook**

Launch:

```bash
jupyter notebook
```

Open `Mining_Customer_Sentiment.ipynb`.

---

# **Saving Interactive Visualizations**

### LDA pyLDAVis:

```python
pyLDAVis.save_html(vis, "lda_vis.html")
```

### BERTopic topics:

```python
topic_model.visualize_topics().write_html("bertopic_topics.html")
```

---

# **License**

MIT License (or specify your preferred license).

---

# **Author**

Christopher Overton
M.S. Data Science Candidate, University of Colorado Boulder


