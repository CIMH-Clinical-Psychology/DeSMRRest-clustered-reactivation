
# Go through all sequence files and check whether they are ok.
# Does each sequence have 16 images?
# Does each sequence consist of unique images?

library(tidyverse)

# Identify sequence files
all_files <- list.files("./graph", full.names = TRUE)
sequence_files <- all_files[grepl("sequence_", all_files)]

# Empty data frame, to be filled with all the sequences
all_files_df <- as.data.frame(matrix(nrow = length(sequence_files), ncol = 16))
names(all_files_df) <- paste0("V", 1:16) # variables will be named V1 - V16 because there is no header in the files

for (file_no in seq_along(sequence_files)) {
  temp_file <- read.table(sequence_files[file_no], header = FALSE, sep = ",")
  all_files_df[file_no, ] <- temp_file 
}

# Make sure that each row (i.e. each sequence file) contains 16 unique images
all_files_df %>% 
  rowwise() %>% # apply to each row
  summarise(unique_count = length(unique(.)), .groups = "drop") %>%  # how many unique elements per row?
  pull(unique_count) %>% 
  all(. == 16) # are there always 16 unique elements?
# -> confirmed!

# Overall: Are there 16 unique elements across all files?
unique(unlist(all_files_df)) %>% length()
# -> confirmed!
