
## 3 Data reading and preprocessing

# Loading packages
pacman::p_load(tidyverse, tidytext, gutenbergr, 
               reshape2, zoo, quanteda, forecast, 
               gridExtra, ggwordcloud, textstem, 
               ggplot2, topicmodels)

book <- gutenberg_download(27780, 
                           mirror = 
                             "http://mirror.csclub.uwaterloo.ca/gutenberg/")

write_csv(book, "Treasure Island.csv")

data <- read.csv("Treasure Island.csv") 


## 3.1 Text preprocessing and chapter division
# Construct chapter numbers
data <- data %>%
  mutate(chapter = cumsum(str_detect(text, 
                                     regex("^chapter", ignore_case = TRUE))))

# Merge each chapter into a complete document
chapter_df <- data %>%
  group_by(chapter) %>%
  summarise(text = paste(text, collapse = " "), .groups = "drop")

# Clean text:
# Replace all punctuation with spaces
# Replace all numbers with spaces
# Convert all text to lowercase
clean_text_df <- chapter_df %>%
  mutate(text = str_replace_all(text, "[[:punct:]]+", " "),
         text = str_replace_all(text, "[[:digit:]]+", " "),
         text = tolower(text))


## 3.2 Data Cleaning 
# 3.2.1 Word frequency statistics of raw data
freq_by_rank_raw <- clean_text_df %>%
  unnest_tokens(word, text) %>%
  count(word, sort = TRUE) %>%
  mutate(rank = row_number(), term_frequency = n / sum(n))

freq_by_rank_raw


# Show the top 20 words
freq_plot <- freq_by_rank_raw %>%
  slice_max(n, n = 20) %>%
  ggplot(aes(x = reorder(word, n), y = n)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(title = "Top 20 Most Frequent Words in Raw Data",
       x = "Word", y = "Frequency") +
  theme_minimal(base_size = 16)
theme(
  plot.title = element_text(size = 18, face = "bold"),
  axis.title.x = element_text(size = 16),
  axis.title.y = element_text(size = 16),
  axis.text = element_text(size = 16)
)
freq_plot

ggsave("1.eps", plot = freq_plot, device = cairo_ps, width = 8, height = 6)

# 3.2.2 Automatically remove stop words
# Load the English stop words provided by tidytext
data(stop_words)

# Filter stop words
freq_by_rank_cleaned <- clean_text_df %>%
  unnest_tokens(word, text) %>%
  anti_join(stop_words, by = "word") %>%
  count(word, sort = TRUE) %>%
  mutate(rank = row_number(),
         term_frequency = n / sum(n))

# Draw a word cloud using tidytext to remove stop words
set.seed(2025)
wordcloud_plot <- ggplot(head(freq_by_rank_cleaned, 100), aes(label = word, size = n)) +
  geom_text_wordcloud() +
  scale_size_area(max_size = 16) +  
  theme_minimal(base_size = 16)

wordcloud_plot
ggsave("2.eps", plot = wordcloud_plot, device = cairo_ps, width = 8, height = 6)

## 3.2.3 Custom stop word processing
# Custom stop words
custom_stop_words <- c("ah", "aye", "hey", "ha", "oh", "ho", 
                       "eh", "just", "like", "still", "back", 
                       "came", "got", "told", "said","well", 
                       "way", "thing", "nothing", "something", 
                       "anything", "everything","im", "ive", 
                       "dont", "youre", "hes", "shes", "weve", 
                       "theyre", "ill", "youll","look", "see", 
                       "run", "walk", "go", "take", "put", "come", 
                       "leave", "begin", "end", "ll.", "ll"
)

# Merge into stop_words data frame
all_stop_words <- bind_rows(
  stop_words,
  tibble(word = custom_stop_words, lexicon = "custom")
)
# Remove stop words and custom stop words
tidy_text <- clean_text_df %>%
  unnest_tokens(word, text) %>%
  anti_join(all_stop_words, by = "word") %>% # Filter standard stop words, all
  count(word, sort = TRUE)

# View high-frequency words after custom stop words filtering
set.seed(2025)
custom_wordcloud_plot <- ggplot(head(tidy_text, 100), aes(label = word, size = n)) +
  geom_text_wordcloud() +
  scale_size_area(max_size = 16) +
  theme_minimal()
custom_wordcloud_plot
ggsave("3.eps", plot = custom_wordcloud_plot, device = cairo_ps, width = 8, height = 6)

## 3.2.4 Add lemmatization
tidy_text <- clean_text_df %>%
  unnest_tokens(word, text) %>%
  mutate(word = lemmatize_words(word)) %>% # Lemmatization
  anti_join(all_stop_words, by = "word") %>%
  count(word, sort = TRUE)

# View the high-frequency words filtered by custom stop words and lemmatization
set.seed(2025)
final_wordcloud_plot <- ggplot(head(tidy_text, 100), aes(label = word, size = n)) +
  geom_text_wordcloud() +
  scale_size_area(max_size = 16) +
  theme_minimal()

ggsave("4.eps", plot = final_wordcloud_plot, device = cairo_ps, width = 8, height = 6)

## 3.2.5 Phase distribution analysis of high-frequency keywords
# Lemmatization + word segmentation + stop word removal + TF-IDF calculation
chapter_words_tfidf <- chapter_df %>%
  unnest_tokens(word, text) %>%
  mutate(word = lemmatize_words(word)) %>% # Lemmatization
  anti_join(all_stop_words, by = "word") %>%
  count(chapter, word, sort = TRUE) %>%
  bind_tf_idf(word, chapter, n)

# Define chapter ranges for each narrative stage
stage_ranges <- list(
  stage1 = 1:6,
  stage2 = 7:12,
  stage3 = 13:24,
  stage4 = 25:34
)

# Labels for each narrative stage
stage_titles <- c(
  stage1 = "Chapters 1–6: Inn & Treasure Map",
  stage2 = "Chapters 7–12: Departure & Conspiracy",
  stage3 = "Chapters 13–24: Island Adventure & Conflict",
  stage4 = "Chapters 25–34: Treasure & Return"
)

# keyword frequency by stage, select top 15
plot_stage <- function(chap_range, title){
  chapter_words_tfidf %>%
    filter(chapter %in% chap_range) %>%
    group_by(word) %>%
    summarize(total_n = sum(n), .groups = "drop") %>%
    slice_max(total_n, n = 15) %>%
    ggplot(aes(x = reorder(word, total_n), y = total_n)) +
    geom_col(fill = "lightblue") +
    coord_flip() +
    labs(title = title, x = "Keyword", y = "Total Frequency") +
    theme_minimal(base_size = 14) +
    theme(
      plot.title = element_text(size = 16, face = "bold"),
      axis.title = element_text(size = 14),
      axis.text = element_text(size = 14)
    )
}

#Generate bar plots
p1 <- plot_stage(stage_ranges$stage1, stage_titles["stage1"])
p2 <- plot_stage(stage_ranges$stage2, stage_titles["stage2"])
p3 <- plot_stage(stage_ranges$stage3, stage_titles["stage3"])
p4 <- plot_stage(stage_ranges$stage4, stage_titles["stage4"])

#Arrange plots into 2×2
p_combined <- grid.arrange(p1, p2, p3, p4, ncol = 1)



# Save each image 
ggsave("5.1.eps", plot = p1, device = cairo_ps, width = 8, height = 6)
ggsave("5.2.eps", plot = p2, device = cairo_ps, width = 8, height = 6)
ggsave("5.3.eps", plot = p3, device = cairo_ps, width = 8, height = 6)
ggsave("5.4.eps", plot = p4, device = cairo_ps, width = 8, height = 6)



## 3.3 LDA Topic Modeling

# Build Document-Term Matrix
dtm <- chapter_words_tfidf %>%
  cast_dtm(document = chapter, term = word, value = n)

# Train LDA model (set the number of topics to 4)
lda_model <- LDA(dtm, k = 4, control = list(seed = 2025))

# Extract β (word-topic distribution)
beta <- tidy(lda_model, matrix = "beta") 
# The probability of each word under each topic


## 3.3.3 Four-stage semantic trend evolution
# Map the chapters into four stages and calculate the average topic distribution of each stage
gamma <- tidy(lda_model, matrix = "gamma")
gamma <- gamma %>%
  mutate(document = as.integer(document)) %>%
  filter(!is.na(document)) %>%
  mutate(
    stage = case_when(
      document %in% 1:6 ~ "Stage 1",
      document %in% 7:12 ~ "Stage 2",
      document %in% 13:24 ~ "Stage 3",
      document %in% 25:34 ~ "Stage 4",
      TRUE ~ NA_character_
    )
  ) %>%
  filter(!is.na(stage))

# Calculate the average γ weight of each stage on each topic
stage_topic_means <- gamma %>% 
  group_by(stage, topic) %>%
  summarize(mean_gamma = mean(gamma), .groups = "drop")

## 3.4 Sentiment Analysis
# Prepare lemmatized word list
lemmatized_words <- tidy_text %>% uncount(n)
total_words <- nrow(lemmatized_words)

# === 1. SentiWordNet ===
data("hash_sentiment_sentiword")
swn_df <- as_tibble(hash_sentiment_sentiword) %>%
  rename(word = x, score = y)
matched_swn <- inner_join(lemmatized_words, swn_df, by = "word")
swn_coverage <- nrow(matched_swn) / total_words

# === 2. Bing ===
bing_df <- get_sentiments("bing")
matched_bing <- inner_join(lemmatized_words, bing_df, by = "word")
bing_coverage <- nrow(matched_bing) / total_words

# === 3. NRC ===
nrc_df <- get_sentiments("nrc") %>%
  filter(sentiment %in% c("positive", "negative"))
matched_nrc <- inner_join(lemmatized_words, nrc_df, by = "word")
nrc_coverage <- nrow(matched_nrc) / total_words

# === 4. SlangSD ===
data("hash_sentiment_slangsd")
slang_df <- hash_sentiment_slangsd %>%
  rename(word = x, score = y)
matched_slang <- inner_join(lemmatized_words, slang_df, by = "word")
slang_coverage <- nrow(matched_slang) / total_words

# === 5. SenticNet ===
data("hash_sentiment_senticnet")
senticnet_df <- hash_sentiment_senticnet %>%
  rename(word = x, score = y)
matched_sentic <- inner_join(lemmatized_words, senticnet_df, by = "word")
senticnet_coverage <- nrow(matched_sentic) / total_words

# === Summarize results into a table ===
coverage_df <- tibble(
  Lexicon = c("SentiWordNet", "Bing", "NRC", "SlangSD", "SenticNet"),
  Coverage = c(swn_coverage, bing_coverage, nrc_coverage, slang_coverage, senticnet_coverage)
)

print(coverage_df)

# === Visualization ===
coverage_plot <- coverage_df %>%
  ggplot(aes(x = Lexicon, y = Coverage, fill = Lexicon)) +
  geom_col(show.legend = FALSE) +
  labs(title = "Coverage Comparison of Sentiment Lexicons",
       x = "Lexicon", y = "Coverage Rate") +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(size = 16, face = "bold"),
    axis.title = element_text(size = 14, face = "bold"),
    axis.text = element_text(size = 13)
  )

coverage_plot

# Save as EPS file
ggsave("13.eps", plot = coverage_plot, device = cairo_ps, width = 8, height = 6)


## 4.2 Keyword time series analysis
# Generate cleaned word frequency data
keyword_trends <- chapter_df %>%
  unnest_tokens(word, text) %>%
  count(chapter, word, sort = TRUE)
all_chapters <- 1:34

# Create island data
island_data <- keyword_trends %>%
  filter(word == "island") %>%
  complete(chapter = all_chapters, fill = list(n = 0)) %>%
  arrange(chapter)

# Time series modeling + fitted extraction
island_ts <- ts(island_data$n)
island_model <- forecast::ses(island_ts)
island_data <- island_data %>%
  mutate(smoothed = as.numeric(fitted(island_model)))

# Drawing
island_plot <- ggplot() +
  geom_line(data = island_data, aes(x = chapter, y = n, color = "Original"), linewidth = 1) +
  geom_line(data = island_data, aes(x = chapter, y = smoothed, color = "Smoothed"), linewidth = 1.2) +
  scale_color_manual(values = c("Original" = "black", "Smoothed" = "orange")) +
  labs(
    title = "Island Frequency Trend Across Chapters",
    x = "Chapter", y = "Frequency", color = "Frequency Type"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(size = 16, face = "bold"),
    axis.title = element_text(size = 14, face = "bold"),
    axis.text = element_text(size = 13),
    legend.title = element_text(size = 13, face = "bold"),
    legend.text = element_text(size = 12)
  )

island_plot

ggsave("7.eps", plot = island_plot, device = cairo_ps, width = 8, height = 6)



#Create ship data
ship_data <- keyword_trends %>%
  filter(word == "ship") %>%
  complete(chapter = all_chapters, fill = list(n = 0)) %>%
  arrange(chapter)

# Create time series and fit
ship_ts <- ts(ship_data$n)
ship_model <- forecast::ses(ship_ts)
ship_data <- ship_data %>% mutate(smoothed = as.numeric(fitted(ship_model)))

# drawing
ship_plot <- ggplot() +
  geom_line(data = ship_data, aes(x = chapter, y = n, color = "Original"), linewidth = 1) +
  geom_line(data = ship_data, aes(x = chapter, y = smoothed, color = "Smoothed"), linewidth = 1.2) +
  scale_color_manual(values = c("Original" = "black", "Smoothed" = "orange")) +
  labs(
    title = "Ship Frequency Trend Across Chapters",
    x = "Chapter", y = "Frequency", color = "Frequency Type"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(size = 16, face = "bold"),
    axis.title = element_text(size = 14, face = "bold"),
    axis.text = element_text(size = 13),
    legend.title = element_text(size = 13, face = "bold"),
    legend.text = element_text(size = 12)
  )

ggsave("8.eps", plot = ship_plot, device = cairo_ps, width = 8, height = 6)

#Create silver data
silver_data <- keyword_trends %>%
  filter(word == "silver") %>%
  complete(chapter = all_chapters, fill = list(n = 0)) %>%
  arrange(chapter)

# Create time series and fit
silver_ts <- ts(silver_data$n)
silver_model <- forecast::ses(silver_ts)
silver_data <- silver_data %>%
  mutate(smoothed = as.numeric(fitted(silver_model)))

# drawing
silver_plot <- ggplot() +
  geom_line(data = silver_data, aes(x = chapter, y = n, color = "Original"), linewidth = 1) +
  geom_line(data = silver_data, aes(x = chapter, y = smoothed, color = "Smoothed"), linewidth = 1.2) +
  scale_color_manual(values = c("Original" = "black", "Smoothed" = "orange")) +
  labs(
    title = "Silver Frequency Trend Across Chapters",
    x = "Chapter", y = "Frequency", color = "Frequency Type"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(size = 16, face = "bold"),
    axis.title = element_text(size = 14, face = "bold"),
    axis.text = element_text(size = 13),
    legend.title = element_text(size = 13, face = "bold"),
    legend.text = element_text(size = 12)
  )

ggsave("9.eps", plot = silver_plot, device = cairo_ps, width = 8, height = 6)

# 4.3.1 Topic keyword distribution
# Extract the top 10 keywords under each topic
top_terms <- beta %>%
  group_by(topic) %>%
  slice_max(beta, n = 10) %>%
  ungroup() %>%
  mutate(term = reorder_within(term, beta, topic))

# Drawing: Keyword distribution map for each topic
topic_plot <- ggplot(top_terms, aes(x = beta, y = term, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  scale_y_reordered() +
  labs(title = "Top 10 keywords for each topic in the LDA model", x = "Probability", y = "Words") +
  theme_minimal(base_size = 17) +  
  theme(
    plot.title = element_text(size = 18, face = "bold"),
    strip.text = element_text(size = 16, face = "bold"),
    axis.title.x = element_text(size = 16, face = "bold"),
    axis.title.y = element_text(size = 16, face = "bold"),
    axis.text.x = element_text(size = 14),
    axis.text.y = element_text(size = 18)
  )
topic_plot
ggsave("10.eps", plot = topic_plot, device = cairo_ps, width = 10, height = 12)


# 4.3.2 Chapter topic weight distribution
# Draw a bar chart
stage_topic_plot <- ggplot(stage_topic_means, aes(x = stage, y = mean_gamma, fill = factor(topic))) +
  geom_col(position = "dodge") +
  labs(
    title = "Average weight of four stages under different themes",
    x = "narrative stage", y = "average γ weight", fill = "topic"
  ) +
  theme_minimal(base_size = 16) +  
  theme(
    plot.title = element_text(size = 18, face = "bold"),
    axis.title = element_text(size = 16, face = "bold"),
    axis.text = element_text(size = 14),
    legend.title = element_text(size = 15, face = "bold"),
    legend.text = element_text(size = 13)
)
stage_topic_plot

ggsave("11.eps", plot = stage_topic_plot, device = cairo_ps, width = 8, height = 6)


## 4.4 Sentiment Analysis
# Load SenticNet lexicon
sentic_df <- as_tibble(hash_sentiment_senticnet) %>%
  rename(word = x, score = y)

# Compute average sentiment score for each chapter
chapter_sentiment_senticnet <- chapter_df %>%
  unnest_tokens(word, text) %>%
  mutate(word = lemmatize_words(word)) %>%
  inner_join(sentic_df, by = "word") %>%
  group_by(chapter) %>%
  summarise(avg_sentiment = mean(score), .groups = "drop")

# Fill in missing chapters
all_chapters <- tibble(chapter = 1:34)
chapter_sentiment_senticnet <- full_join(all_chapters, chapter_sentiment_senticnet, by = "chapter") %>%
  mutate(avg_sentiment = replace_na(avg_sentiment, 0))

# Apply exponential smoothing
sentiment_ts <- ts(chapter_sentiment_senticnet$avg_sentiment)
smoothed_model <- forecast::ses(sentiment_ts)
chapter_sentiment_senticnet <- chapter_sentiment_senticnet %>%
  mutate(smoothed = as.numeric(fitted(smoothed_model)))

# Plot emotional arc
sentiment_plot <- ggplot(chapter_sentiment_senticnet, aes(x = chapter)) +
  geom_line(aes(y = avg_sentiment, color = "Original"), linewidth = 1) +
  geom_line(aes(y = smoothed, color = "Smoothed"), linewidth = 1.2) +
  scale_color_manual(values = c("Original" = "black", "Smoothed" = "orange")) +
  labs(
    title = "Emotional Arc of Treasure Island (Using SenticNet)",
    x = "Chapter", y = "Average Sentiment Score", color = "Line Type"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(size = 16, face = "bold"),
    axis.title = element_text(size = 14, face = "bold"),
    axis.text = element_text(size = 13),
    legend.title = element_text(size = 13, face = "bold"),
    legend.text = element_text(size = 12)
  )

sentiment_plot

# Save sentiment plot as EPS
ggsave("14.eps", plot = sentiment_plot, device = cairo_ps, width = 8, height = 6)













