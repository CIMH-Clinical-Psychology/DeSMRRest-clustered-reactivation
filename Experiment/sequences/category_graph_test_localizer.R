
# Go through all localizer files and check whether they are ok.

library(tidyverse)

# Identify localizer files
all_files <- list.files("./sequences/category_graph", full.names = TRUE)
localizer_files <- all_files[grepl("localizer", all_files)]

# There are five files per participant. Identify participant IDs.
participants <- str_extract(localizer_files, "[0-9]+") # first occurrence of digits
participants <- sort(unique(as.numeric(participants)))

# Set up a data frame that stores files which are problematic
problematic_files <- as.data.frame(matrix(nrow = 0, ncol = 5))
names(problematic_files) <- c("participant", "file_no", "stimulus", "type", "count")

for (file_idx in seq_along(localizer_files)) {

  temp_file_name <- localizer_files[file_idx]
  
  # skip file for participant 0
  if (grepl('participant_0', temp_file_name)){
    next
  }
  
  temp_file <- read.csv(temp_file_name, header = FALSE)
  names(temp_file) <- c("img", "square")
  
  temp_participant <- str_extract(temp_file_name, "[0-9]+")
  temp_file_no <- str_extract(temp_file_name, "[0-9].csv")
  temp_file_no <- sub(".csv", "", temp_file_no)
  
  counts <- table(temp_file$img)
  
  # If each image doesn't occur 27 times, save file as problematic.
  if (!all(counts == 27)) {
    problematic_stim <- names(counts)[counts != 27]
    problematic_counts <- counts[counts != 27]
    
    temp_problematic_df <- data.frame(
      participant = temp_participant,
      file_no = temp_file_no,
      stimulus = problematic_stim,
      type = "img_count",
      count = problematic_counts
    )
    
    problematic_files <- rbind(problematic_files, temp_problematic_df)
  }
  
  # Count whether images appear equally often with/without a square
  counts <- table(temp_file$square, temp_file$img)
  
  # All non-square images need to appear 24 times
  if (!all(counts[1, ] == 24)) {
    problematic_stim <- names(counts[1, ])[counts[1, ] != 24]
    problematic_counts <- as.vector(counts[1, ][counts[1, ] != 24])
    
    temp_problematic_df <- data.frame(
      participant = temp_participant,
      file_no = temp_file_no,
      stimulus = problematic_stim,
      type = "no_square",
      count = problematic_counts
    )
    
    problematic_files <- rbind(problematic_files, temp_problematic_df)
  }
  
  # All square images need to appear 3 times
  if (!all(counts[2, ] == 3)) {
    problematic_stim <- names(counts[2, ])[counts[2, ] != 3]
    problematic_counts <- as.vector(counts[2, ][counts[2, ] != 3])
    
    temp_problematic_df <- data.frame(
      participant = temp_participant,
      file_no = temp_file_no,
      stimulus = problematic_stim,
      type = "square",
      count = problematic_counts
    )
    
    problematic_files <- rbind(problematic_files, temp_problematic_df)
  }
  
}

# Problematic files.
# participant: which participant
# file_no: which file
# e.g. participant_39_5.csv

# stimulus: which img number is the problematic one? E.g. img 12
# type: Which error type is the problem.
#   - `no_square` means that the number of occurrences where there was no square is wrong (i.e. more or less than
#     14 occurences of an image without a square).
#   - `square` means that the number of squares per image is wrong (i.e. more or less than 1 square per image)
#   - `img_count` means that the overall number of image occurrences is wrong (i.e. an image was presented more
#      or less than 15 times)

# No problematic files detected
problematic_files %>% view()
