
# Go through all learning unit files and check whether they are ok.

library(tidyverse)

##################################
## EXTRACT RELEVANT INFORMATION ##
##################################

# Identify learning unit files
all_files <- list.files("./category_graph", full.names = TRUE)
learning_unit_files <- all_files[grepl("learning_units", all_files)]
learning_unit_files <- learning_unit_files[-length(learning_unit_files)] # take out last test file

# Sequence of letters
full_seq <- c("ABCDEFGEHIBJ")
full_seq <- unlist(strsplit(full_seq, ""))
# Add first letters at the end to have a "full circle"
long_seq <- paste0(c(full_seq, "A", "B"), collapse = "")

# Generate all possible sequence triplets.
all_triplets <- as.data.frame(matrix(nrow = 12, ncol = 3))
names(all_triplets) <- paste0("item", 1:3)

for (i in seq_along(full_seq)) {
  if (i == 1) {
    all_triplets[i, ] <- full_seq[1:3]
  } else {
    all_triplets[i, ] <- full_seq[c(i:length(full_seq), 1:(i - 1))][1:3]
  }
}

# Unite triplets so we get a unique string for each
all_triplets_unite <- all_triplets %>% 
  unite("triplet", paste0("item", 1:3), sep = "")

# Preallocate empty lists and vectors to store test results
# Check whether triplets are fine
check_all_triplets <- vector(length = length(learning_unit_files))
# Check how many times the correct choice is in a given position
correct_table <- list()
# Check how many times the correct choice is in the same position in a row
max_correct_in_a_row <- list()
# Check whether the correct item is actually the correct one.
check_correct <- vector(length = length(learning_unit_files))

# Loop through all the files
for (file in seq_along(learning_unit_files)) {
  
  ## READ DATA AND RESTRUCTURE ##
  # Read in file
  learning_units <- read.csv(learning_unit_files[file], skip = 1, header = FALSE, sep = ";")
  
  # Structure: item1, item2, choice1, choice2, choice3, correct choice
  
  # Bring into an "R-friendly" format. Label each column in a meaningful way, with the unit number at the end.
  # E.g. for unit 3, we'd have item1_3, item2_3, choice1_3, choice2_3, choice3_3, correct_3.
  
  # First unite all the columns, then separate them with a comma
  units_df <- 
    learning_units %>% 
    unite("x", sep = ",") %>% 
    separate(
      x, 
      sep = ",",
      paste0(
        rep(c("item1_", "item2_", "choice1_", "choice2_", "choice3_", "correct_"), 12), 
        rep(1:12, each = 6)
      )
    )
  
  ## CHECK TRIPLETS ##
  # The goal is to compare the existing triplets in the sequences with all possible triplets.
  # For this, we extract item1, item2 and the correct choice for each unit.
  # Initiate an empty list.
  triplet_list <- list()
  
  # Loop through the rows of the units dataframe, i.e. through all the runs one participant completes. We need to
  # treat each of these separately, i.e. in each run, each sequence has to be present.
  for(temp_sequence in 1:nrow(units_df)) {
    temp_df <- data.frame(unit = 1:12,
                          item1 = NA,
                          item2 = NA,
                          item3 = NA)
    
    # Go through each unit ...
    for (unit in 1:12) {
      # pick item 1
      temp_df$item1[unit] <- units_df[[paste0("item1_", unit)]][temp_sequence]
      # pick item 2
      temp_df$item2[unit] <- units_df[[paste0("item2_", unit)]][temp_sequence]
      # pick the choice that is the correct one
      temp_df$item3[unit] <- units_df[[paste0("choice", units_df[[paste0("correct_", unit)]][temp_sequence], "_", unit)]][temp_sequence]
    }
    
    triplet_list[[temp_sequence]] <- temp_df
  }
  
  # We unite the triplets again to have unique strings for each of them.
  triplet_list_unite <- lapply(triplet_list, function(x) {
    x <- x %>% 
      unite("triplet", paste0("item", 1:3), sep = "")
  })
  
  # Check whether each run in triplet_list contains all possible triplets
  check_triplets <- lapply(triplet_list_unite, function(x){
    all(x$triplet %in% all_triplets_unite$triplet)
  })
  
  # If this is check true, each sequence in a given file contains each triplet
  check_all_triplets[file] <- all(unlist(check_triplets))
  
  # CHECK DISTRIBUTION OF CORRECT CHOICES ##
  correct <- units_df %>% 
    select(starts_with("correct_"))
  
  correct_table[[file]] <- apply(correct, 1, table)
  in_a_row <- apply(correct, 1, rle)
  in_a_row <- lapply(in_a_row, function(x){
    as.vector(x$lengths)
  })
  max_correct_in_a_row[[file]] <- unlist(lapply(in_a_row, max))
  
  ## CHECK WHETHER CORRECT CHOICE IS ACTUALLY CORRECT ##
  # Bring into long format for easy comparison
  units_lf <- units_df %>% 
    mutate(sequence_no = 1:nrow(.)) %>% 
    pivot_longer(cols = -sequence_no,
                 names_to = c(".value", "triplet_no"), values_to = "letter",
                 names_pattern = "(.*)_(.*)")
  
  # Bind together letter pairs (item1 and item2)
  units_lf <- units_lf %>% 
    mutate(letter_pair = paste0(item1, item2))
  
  # Locate the letter pairs in the string
  units_lf <- units_lf %>% 
    mutate(location1 = str_locate(long_seq, letter_pair)[, 1],
           location2 = str_locate(long_seq, letter_pair)[, 2])
  
  # Location 3 has to be 1 after location 2
  units_lf <- units_lf %>% 
    mutate(location3 = location2 + 1)
  
  # Which letter corresponds to position 3?
  units_lf <- units_lf %>% 
    group_by(sequence_no, triplet_no) %>% 
    mutate(correct_letter = substr(long_seq, location3, location3)) %>% 
    ungroup()
  
  # Which letter is shown as the correct one?
  units_lf <- units_lf %>% 
    mutate(shown_as_correct = case_when(
      correct == 1 ~ choice1,
      correct == 2 ~ choice2,
      correct == 3 ~ choice3
    ))
  
  units_lf <- units_lf %>% 
    mutate(check = correct_letter == shown_as_correct)
  
  check_correct[file] <- all(units_lf$check)
}

################
## RUN CHECKS ##
################

# For each learning unit, all triplets appear
all(check_all_triplets) # This must be TRUE

# Distribution of correct choices.
# List: File, in the order of learning_unit_files
# Columns: Run 1 - 15
# Rows: Choice 1 - 3
correct_table

# Maximum number of occurrences of correct position within a given sequence.
# List: File, in the order of learning_unit_files
# I.e. for file 1, the maximum number of the same choice position in a row is 2 for run 1.
# For file 5, the maximum number of the same choice position in a row is 7 for run 8.
max_correct_in_a_row

# highest number of repetitions:
max(unlist(max_correct_in_a_row))

# Correct positions are always correct
all(check_correct)
