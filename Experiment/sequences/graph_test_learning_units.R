
# Go through all learning unit files and check whether they are ok.

library(tidyverse)

# Identify learning unit files
all_files <- list.files("./graph", full.names = TRUE)
learning_unit_files <- all_files[grepl("learning_units", all_files)]
learning_unit_files <- learning_unit_files[-length(learning_unit_files)] # take out last test file

# Sequence of letters
full_seq <- c("ABCDEBFGHIFJKLMJANOP")
full_seq <- unlist(strsplit(full_seq, ""))

# Preallocate empty lists and vectors to store test results
check_all_triplets <- vector(length = length(learning_unit_files))
correct_table <- list()
max_correct_in_a_row <- list()

# Loop through all the files
for (file in seq_along(learning_unit_files)) {
  # Read in file
  learning_units <- read.csv(learning_unit_files[file], skip = 1, header = FALSE, sep = ";")
  
  # Structure: item1, item2, choice1, choice2, choice3, correct choice
  
  # Bring into an "R-friendly" format. Label each column in a meaningful way, with the unit number at the end.
  # E.g. for unit 3, we'd have item1_3, item2_3, choice1_3, choice2_3, choice3_3, correct_3.
  # First, set up an empty data frame with the desired column names:
  units_df <- as.data.frame(matrix(nrow = nrow(learning_units), ncol = 20 * 6))
  new_names <- paste0(rep(c("item1_", "item2_", "choice1_", "choice2_", "choice3_", "correct_"), 20), rep(1:20, each = 6))
  names(units_df) <- new_names
  
  # Split up the sequences in the dataframe we read in. Each cell of the dataframe will now contain a list.
  split_df <- learning_units %>% 
    mutate(across(everything(), ~strsplit(., ",")))
  
  # Fill the pre-allocated dataframe with the elements of the split list.
  # I.e. item1 is the first item of the list, item2 is the second item of the list, choice1 is the third item of the list.
  # Loop through the units (indicated behind the _ in the column name)
  for (unit in 1:20) {
    units_df[[paste0("item1_", unit)]] <- unlist(lapply(split_df[[paste0("V", unit)]], `[`, 1))
    units_df[[paste0("item2_", unit)]] <- unlist(lapply(split_df[[paste0("V", unit)]], `[`, 2))
    units_df[[paste0("choice1_", unit)]] <- unlist(lapply(split_df[[paste0("V", unit)]], `[`, 3))
    units_df[[paste0("choice2_", unit)]] <- unlist(lapply(split_df[[paste0("V", unit)]], `[`, 4))
    units_df[[paste0("choice3_", unit)]] <- unlist(lapply(split_df[[paste0("V", unit)]], `[`, 5))
    units_df[[paste0("correct_", unit)]] <- unlist(lapply(split_df[[paste0("V", unit)]], `[`, 6))
  }
  
  # Generate all possible sequence triplets. Start with an empty dataframe that has each triplet item in a separate
  # column.
  all_triplets <- data.frame(triplet_no = seq_along(full_seq),
                             item1 = NA,
                             item2 = NA,
                             item3 = NA)
  
  # Fill the dataframe by looping through the full sequence (set up at the top of the script).
  for (i in seq_along(full_seq)) {
    all_triplets$item1[i] <- full_seq[0 + i]
    # Special case: If the index exceeds 20 (i.e. the number of items in the sequence),
    # take the modulo, i.e. take the first item again as there is no 21st.
    all_triplets$item2[i] <- full_seq[ifelse((1 + i) <= 20, (1 + i), (1 + i) %% 20)]
    all_triplets$item3[i] <- full_seq[ifelse((2 + i) <= 20, (2 + i), (2 + i) %% 20)]
  }
  
  # To compare the possible triplets with the existing ones, we need to reduce the file we read in. We only
  # want item1, item2 and then the correct choice. Initiate an empty list.
  triplet_list <- list()
  
  # Loop through the rows of the units dataframe, i.e. through all the runs one participant completes. We need to
  # treat each of these separately, i.e. in each run, each sequence has to be present.
  for(temp_sequence in 1:nrow(units_df)) {
    temp_df <- data.frame(unit = 1:20,
                          item1 = NA,
                          item2 = NA,
                          item3 = NA)
    
    # For every sequence, go through each unit and within that unit, pick item1, item2 and the choice item that is
    # identified as correct in the respetive column.
    for (unit in 1:20) {
      temp_df$item1[unit] <- units_df[[paste0("item1_", unit)]][temp_sequence]
      temp_df$item2[unit] <- units_df[[paste0("item2_", unit)]][temp_sequence]
      temp_df$item3[unit] <- units_df[[paste0("choice", units_df[[paste0("correct_", unit)]][temp_sequence], "_", unit)]][temp_sequence]
    }
    
    triplet_list[[temp_sequence]] <- temp_df
  }
  
  # We unite the triplets again to have unique strings for each of them. We do this for both the
  # possible triplets and the triplets existing in the file.
  triplet_list_unite <- lapply(triplet_list, function(x) {
    x <- x %>% 
      unite("triplet", paste0("item", 1:3), sep = "")
  })
  
  all_triplets_unite <- all_triplets %>% 
    unite("triplet", paste0("item", 1:3), sep = "")
  
  # Check whether each dataframe in triplet_list contains all the triplets in all_triplets
  check_triplets <- lapply(triplet_list_unite, function(x){
    all(x$triplet %in% all_triplets_unite$triplet)
  })
  
  # If this is check true, each sequence contains each triplet
  check_all_triplets[file] <- all(unlist(check_triplets))
  
  # Distribution of correct items
  correct <- units_df %>% 
    select(starts_with("correct_"))
  
  correct_table[[file]] <- apply(correct, 1, table)
  in_a_row <- apply(correct, 1, rle)
  in_a_row <- lapply(in_a_row, function(x){
    as.vector(x$lengths)
  })
  max_correct_in_a_row[[file]] <- unlist(lapply(in_a_row, max))
  
}

#### CHECKS ####


# For each learning unit, all triplets appear
all(check_all_triplets) # This must be TRUE

# Distribution of correct choices.
# List: File, in the order of
learning_unit_files
# Columns: Run 1 - 15
# Rows: Choice 1 - 3
correct_table

# Maximum number of occurences of correct position within a given sequence.
# List: File, in the order of
learning_unit_files
# I.e. for file 1, the maximum number of the same choice position in a row is 2 for run 1.
# For file 5, the maximum number of the same choice position in a row is 7 for run 8.
max_correct_in_a_row


#### CHECK CORRECT ####
# Check whether the correct item is actually the correct one.
check_correct <- vector(length = length(learning_unit_files))

# Add first letters at the end to have a "full circle"
long_seq <- paste0(c(full_seq, "A", "B"), collapse = "")

for (file in seq_along(learning_unit_files)) {
  # Read in file
  learning_units <- read.csv(learning_unit_files[file], skip = 1, header = FALSE, sep = ";")
  
  # Restructuring: Same as above
  units_df <- as.data.frame(matrix(nrow = nrow(learning_units), ncol = 20 * 6))
  new_names <- paste0(rep(c("item1_", "item2_", "choice1_", "choice2_", "choice3_", "correct_"), 20), rep(1:20, each = 6))
  names(units_df) <- new_names
  
  split_df <- learning_units %>% 
    mutate(across(everything(), ~strsplit(., ",")))
  
  for (unit in 1:20) {
    units_df[[paste0("item1_", unit)]] <- unlist(lapply(split_df[[paste0("V", unit)]], `[`, 1))
    units_df[[paste0("item2_", unit)]] <- unlist(lapply(split_df[[paste0("V", unit)]], `[`, 2))
    units_df[[paste0("choice1_", unit)]] <- unlist(lapply(split_df[[paste0("V", unit)]], `[`, 3))
    units_df[[paste0("choice2_", unit)]] <- unlist(lapply(split_df[[paste0("V", unit)]], `[`, 4))
    units_df[[paste0("choice3_", unit)]] <- unlist(lapply(split_df[[paste0("V", unit)]], `[`, 5))
    units_df[[paste0("correct_", unit)]] <- unlist(lapply(split_df[[paste0("V", unit)]], `[`, 6))
  }
  
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

# Correct positions are always correct
all(check_correct)

