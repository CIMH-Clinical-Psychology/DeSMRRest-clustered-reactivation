
# Go through all sequence files and check whether they are ok.
# Does each sequence have 16 images?
# Does each sequence consist of unique images?

library(tidyverse)

sequence_length <- 10

# Identify sequence files
# Juli: I changed the path because my project is in the main folder
all_files <- list.files("./category_graph", full.names = TRUE)
sequence_files <- all_files[grepl("sequence_", all_files)]

# Read in each file using map and then bind all of them together in one data frame.
# Each row is one sequence, each column is one position in the sequence
all_sequences <- 
  sequence_files %>%
  map(read.table, header = FALSE, sep = ",") %>%
  reduce(rbind)  

# Does each sequence consist of unique elements?
unique_counts <- apply(all_sequences, 1, function(x) length(unique(x)))
all(unique_counts == sequence_length)
# -> confirmed!

# Overall: Are there 10 unique elements across all files?
# I.e. make sure there are 10 different images, not more or less
unique(unlist(all_sequences)) %>% length()
# -> confirmed!
