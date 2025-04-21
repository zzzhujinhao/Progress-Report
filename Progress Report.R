
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

theme_set(theme_bw())

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


## 3.2 Word frequency and TF-IDF extraction
# 3.2.1 Word frequency statistics of raw data
freq_by_rank_raw <- clean_text_df %>%
  unnest_tokens(word, text) %>%
  count(word, sort = TRUE) %>%
  mutate(rank = row_number(), term_frequency = n / sum(n))

freq_by_rank_raw


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
ggplot(head(freq_by_rank_cleaned, 100), aes(label = word, size = n)) +
  geom_text_wordcloud() +
  scale_size_area(max_size = 10) +
  theme_minimal()


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
                       "leave", "begin", "end","hand", "head", 
                       "cry", "word", "fall", "dead"
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
ggplot(head(tidy_text, 100), aes(label = word, size = n)) +
  geom_text_wordcloud() +
  scale_size_area(max_size = 10) +
  theme_minimal()


## 3.2.4 Add lemmatization
tidy_text <- clean_text_df %>%
  unnest_tokens(word, text) %>%
  mutate(word = lemmatize_words(word)) %>% # Lemmatization
  anti_join(all_stop_words, by = "word") %>%
  count(word, sort = TRUE)

# View the high-frequency words filtered by custom stop words and lemmatization
set.seed(2025)
ggplot(head(tidy_text, 100), aes(label = word, size = n)) +
  geom_text_wordcloud() +
  scale_size_area(max_size = 10) +
  theme_minimal()


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
    geom_col(fill = "darkorange") +
    coord_flip() +
    labs(title = title, x = "Keyword", y = "Total Frequency") +
    theme_minimal()
}

#Generate bar plots
p1 <- plot_stage(stage_ranges$stage1, stage_titles["stage1"])
p2 <- plot_stage(stage_ranges$stage2, stage_titles["stage2"])
p3 <- plot_stage(stage_ranges$stage3, stage_titles["stage3"])
p4 <- plot_stage(stage_ranges$stage4, stage_titles["stage4"])

#Arrange plots into 2×2
gridExtra::grid.arrange(p1, p2, p3, p4, ncol = 2)


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


## 4.2 TF-IDF 提取结果：识别章节特有关键词
# 选取想展示的章节
selected_chapters <- c(6, 15, 28)

chapter_words_tfidf %>%
  filter(chapter %in% selected_chapters) %>%
  group_by(chapter) %>%
  slice_max(tf_idf, n = 10) %>%
  ungroup() %>%
  ggplot(aes(x = tf_idf, y = reorder_within(word, tf_idf, chapter), fill = as.factor(chapter))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~chapter, scales = "free") +
  scale_y_reordered() +
  labs(
    title = "TF-IDF Top10 keywords in each chapter",
    x = "TF-IDF",
    y = "Keyword"
  ) +
  theme_minimal()


## 4.3 Keyword time series analysis
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
ggplot(island_data, aes(x = chapter)) +
  geom_line(aes(y = n), color = "black", linewidth = 0.7) +
  geom_line(aes(y = smoothed), color = "blue", linewidth = 1) +
  labs(
    title = "island frequency trend chart with chapters",
    x = "Chapter", y = "Frequency"
  ) +
  theme_minimal()


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
ggplot(ship_data, aes(x = chapter)) +
  geom_line(aes(y = n), color = "black", linewidth = 0.7) +
  geom_line(aes(y = smoothed), color = "orange", linewidth = 1) +
  labs(
    title = "ship frequency trend chart with chapters",
    x = "Chapter", y = "Frequency"
  ) +
  theme_minimal()



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
ggplot(silver_data, aes(x = chapter)) +
  geom_line(aes(y = n), color = "black", linewidth = 0.7) +
  geom_line(aes(y = smoothed), color = "purple", linewidth = 1) +
  labs(
    title = "silver frequency trend chart with chapters",
    x = "Chapter", y = "Frequency"
  ) +
  theme_minimal()


# 4.4.1 Topic keyword distribution
# Extract the top 10 keywords under each topic
top_terms <- beta %>%
  group_by(topic) %>%
  slice_max(beta, n = 10) %>%
  ungroup() %>%
  mutate(term = reorder_within(term, beta, topic))

# Drawing: Keyword distribution map for each topic
ggplot(top_terms, aes(x = beta, y = term, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  scale_y_reordered() +
  labs(title = "Top 10 keywords for each topic in the LDA model", x = "Probability", y = "Words") +
  theme_minimal()

# 4.4.2 Chapter topic weight distribution
# Draw a bar chart
ggplot(stage_topic_means, aes(x = stage, y = mean_gamma, fill = factor(topic))) +
  geom_col(position = "dodge") +
  labs(
    title = "Average weight of four stages under different themes",
    x = "narrative stage", y = "average γ weight", fill = "theme"
  ) +
  theme_minimal()



