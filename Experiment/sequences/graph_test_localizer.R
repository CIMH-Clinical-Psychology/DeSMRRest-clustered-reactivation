
# Go through all localizer files and check whether they are ok.

library(tidyverse)

# Identify localizer files
all_files <- list.files("./graph", full.names = TRUE)
localizer_files <- all_files[grepl("localizer", all_files)]

# There are five files per participant. Identify participant IDs.
participants <- str_extract(localizer_files, "[0-9]+") # first occurence of digits
participants <- sort(unique(as.numeric(participants)))

problematic_files <- as.data.frame(matrix(nrow = 0, ncol = 5))
names(problematic_files) <- c("participant", "file_no", "stimulus", "type", "count")

for (file_idx in seq_along(localizer_files)) {
  # Good thing is: It separates the file by the . automatically, so we have the IMG number
  # and whether there was a square or not
  temp_file_name <- localizer_files[file_idx]
  if (grepl('participant_0', temp_file_name)){
    next
  }
  temp_file <- read.csv(temp_file_name, header = FALSE)
  names(temp_file) <- c("img", "square")
  
  temp_participant <- str_extract(temp_file_name, "[0-9]+")
  temp_file_no <- str_extract(temp_file_name, "[0-9].csv")
  temp_file_no <- sub(".csv", "", temp_file_no)
  
  counts <- table(temp_file$img)
  
  if (!all(counts == 15)) {
    problematic_stim <- names(counts)[counts != 15]
    problematic_counts <- counts[counts != 15]
    
    temp_problematic_df <- data.frame(
      participant = temp_participant,
      file_no = temp_file_no,
      stimulus = problematic_stim,
      type = "img_count",
      count = problematic_counts
    )
    
    problematic_files <- rbind(problematic_files, temp_problematic_df)
  }
  
  counts <- table(temp_file$square, temp_file$img)
  
  if (!all(counts[1, ] == 14)) {
    problematic_stim <- names(counts[1, ])[counts[1, ] != 14]
    # UNTIL HERE
    problematic_counts <- as.vector(counts[1, ][counts[1, ] != 14])
    
    temp_problematic_df <- data.frame(
      participant = temp_participant,
      file_no = temp_file_no,
      stimulus = problematic_stim,
      type = "no_square",
      count = problematic_counts
    )
    
    problematic_files <- rbind(problematic_files, temp_problematic_df)
  }
  
  if (!all(counts[2, ] == 1)) {
    problematic_stim <- names(counts[2, ])[counts[2, ] != 1]
    # UNTIL HERE
    problematic_counts <- as.vector(counts[2, ][counts[2, ] != 1])
    
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
#   - `no_square` means that the number of occurences where there was no square is wrong (i.e. more or less than
#     14 occurences of an image without a square).
#   - `square` means that the number of squares per image is wrong (i.e. more or less than 1 square per image)
#   - `img_count` means that the overall number of image occurences is wrong (i.e. an image was presented more
#      or less than 15 times)
problematic_files %>% view()

# Overall, (almost) all files for participants 21 - 40 are off
problematic_files %>% 
  select(participant, file_no) %>% 
  unique()

