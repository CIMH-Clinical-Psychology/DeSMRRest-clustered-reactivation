scenario = "post_test_parallel";
$write_codes = false; # needs to be disabled if not using the MEG computer
$exp_mode = "category_graph"; # which sequence mode is used: graph or category or category_graph
active_buttons = 3;
response_logging = log_active;
response_matching = simple_matching;
default_font_size = 26;
default_font = "Arial";
response_port_output = false; # do not write button codes to MEG trigger port
write_codes = $write_codes; # dont change this, change the SDL var above


###############################################################
## Port Trigger Table
## Port Code | Meaning
## ---------------------------------------------------
## 0      | don't send trigger
## 1-16   | item that is being shown
## 32		 | the feedback prompt
## 99		 | fixation cross before first image
## 255    | start and end trigger
###############################################################

# All settings are stored and loaded via PCL in settings.pcl
# this file is more or less a copy of 2_graph_learning.
# however, I think it is cleaner to have a separate script,
# and the main difference is that in the post-test there is no feedback given.
begin;

# define elements that we use later
# sizes are not defined here but in settings.pcl
box {color=0,200,255; height=1; width=1;} box_selected;
box {color=0,0,0;    height=1; width=1;} box_black;
text {caption="+";font_color=150,150,150;}text_trial_start;
picture {background_color =0,0,0; default_code = "default_background"; } default;


# things that are declared but never used (but need to be present due to PCL/SDL restrictions).
# these objects are called in the set_seq in function.pcl, but they are not used within
# the testing script. they are dummy-defined here, but never actually appear.
# however, if we do not define them, presentation will give an error.
# you can safely ignore them.
array{bitmap{filename="images\\dummy.jpg";};}training_items;
box {height=1; width=1;} box_red;
box {height=1; width=1;} box_green;
trial{stimulus_event{picture{text{caption="";};x=0;y=0;text{caption="";};x=0;y=0;text{caption="";};x=0;y=0;text{caption="";};x=0;y=0;
	text{caption="";};x=0;y=0;text{caption="";};x=0;y=0;text{caption="";};x=0;y=0;text{caption="";};x=0;y=0;text{caption="";};x=0;y=0;
} picture_feedback;}stim_feedback;};


picture {# the picture with the first item of the 3-part sequence
        # size, position and content are set dynamically
	default_code = "seq1";
	bitmap{filename="images\\dummy.jpg";}; # item1
	x = 0;	y = 0;
} picture_seq1;

picture {# the second item of the 3-part excerpt
        # size, position and content are set dynamically
	default_code = "seq2";
	bitmap{filename="images\\dummy.jpg";}; # item1
	x = 0;	y = 0;
	bitmap{filename="images\\dummy.jpg";}; # item2
	x = 0;	y = 0;
} picture_seq2;

picture {
	default_code = "fixation";
	text{caption="+";} text_fixation_cross;
	x = 0; y = 0;
}picture_fixation;


picture{
	default_code = "choice";
	# sequence item 1
	bitmap{filename="images\\dummy.jpg";};x = 0;	y = 0; #item1
	# sequence item 2
	bitmap{filename="images\\dummy.jpg";};x = 0;	y = 0; #item2
	
	# sequence 3 choice items
	bitmap{filename="images\\dummy.jpg";};x = 0;y = 0; #item3
	bitmap{filename="images\\dummy.jpg";};x = 0;y = 0; #item3
	bitmap{filename="images\\dummy.jpg";};x = 0;y = 0; #item3
	
	# these are defined, but they are not used due to SDL restrictions of reusing code
	text{caption="";};x = 0;y = 0;
	text{caption="";};x = 0;y = 0;
	text{caption="";};x = 0;y = 0;
	text{caption="";};x = 0;y = 0;
	text {caption = "?";font_color=150, 150, 150;};x=0;y=0;
} picture_prompt;


picture{ # will be shown shortly to highlight the selected item
         # basically the same as picture_prompt but with a blue frame around selected
		 # all values are set dynamically
	default_code = "selected";
	bitmap{filename="images\\dummy.jpg";}; #item1
	x = 0;	y = 0;
	bitmap{filename="images\\dummy.jpg";}; #item2
	x = 0;	y = 0;
	
   box { height = 1; width = 1;}; x = 0; y = 0;	on_top = false; #possible frame around item
	bitmap {filename="images\\dummy.jpg";};	x = 0;	y = 0; #item3
	
   box { height = 1; width = 1;}; x = 0; y = 0;	on_top = false; #possible frame around item
	bitmap{filename="images\\dummy.jpg";};x = 0;	y = 0; #item3
	
   box { height = 1; width = 1;}; x = 0; y = 0;	on_top = false; #possible frame around item
	bitmap{filename="images\\dummy.jpg";};	x = 0;	y = 0; #item3

	text {caption = "";};	x = 0;	y = -250;
} picture_selected;



####################################
### Trial Screens ##################
####################################

trial { # Welcome screen
	picture{
		default_code = "welcome screen";
		text{caption="Loading experiment in mode $exp_mode. \nPlease wait...";};
		x = 0;
		y = 0;
   };
	time = 0;
	port_code = 255;
	code = "Start trigger (255)";
	duration = 2000;
	

	picture{
		default_code = "welcome screen";
		text{caption="Willkommen zum letzten Teil des Experiments!
		
Bitte drücke einen Taste, um zu beginnen.
		";};
		x = 0;
		y = 0;
   };
	time = 2000;
	duration = response;
	picture{
		text{caption="Dir werden jetzt noch einmal die Objekte wie in der Lerneinheit gezeigt,
deine Aufgabe ist es, richtig zu benennen, welches Objekt als nächstes folgt.

Du bekommst diesmal kein Feedback, ob Du richtig oder falsch lagst.
Diesmal gibt es kein Zeitlimit zum antworten. 
Wenn du eine Antwort nicht weißt, drücke eine beliebige Taste.

						  Bitte drücke eine Taste, um fortzufahren.   ";};

		x = 0;
		y = 0;
	};
    duration = response;

	picture{
		text{caption="Bitte drücke eine Taste, um zu beginnen.";};x = 0;y = 0;
	};
   duration = response;
} screen_welcome;


## main trial
# this trial defines one sequence of 2 items + a choice of 3 following
trial { # img1, img2 and choice of 3 other imgs, inbetween a fixation cross
	stimulus_event{picture{text text_trial_start;x = 0;y = 0;} picture_training_pre; code = "fixation (99)";port_code=99;duration=next_picture;
		}stim_preq;
	stimulus_event{picture picture_seq1;code = "seq1 (1)";	duration=next_picture; port_code=1;
		}stim_seq1;
	stimulus_event{ picture picture_seq2;code = "seq2 (2)"; duration=next_picture; port_code=2;
		}stim_seq2;
} trial_seq;

trial{ # present three options here
	trial_type=first_response;
	stimulus_event{picture picture_prompt;code = "choice (3)"; duration = next_picture; port_code=3;
		}stim_prompt;
} trial_prompt;

trial { # highlight only the selection. Do not highlight the correct one.
	    # stim_feedback gives feedback on which button has been pressed.
    stimulus_event{picture picture_selected; code="highlight (4)"; port_code=32;}stim_selected;
}trial_selected;


trial{
	picture{text{caption="Ende des Experimentteils.\nBitte drücke eine Taste und melde dich beim Experimentleiter.";}text_final;x=0;y=0;};
	duration=response;
	picture{text{caption="";};x=0;y=0;
	};
	port_code = 255;
	code = "End trigger (255)";
	duration=250;
}trial_final;


trial { # warning in case no participant number is set
	picture{text {caption="!!! WARNUNG !!!\n\nKeine Participant-Nummer gesetzt.\n\nBitte diese Fehlermeldung dem Experimentleiter mitteilen.";};x=0;y=0;};
	code = "WARNING1";
	duration=response;
}trial_warning_participant_number;

trial { # warning in case write_codes is false which means no MEG triggers are sent
	picture{text {caption="!!! WARNUNG !!!\n\nwrite_codes=false.\n\nBitte diese Fehlermeldung dem Experimentleiter mitteilen.";};x=0;y=0;};
	code = "WARNING2";
	duration=response;
}trial_warning_write_codes;


#############################################
###### PCL ##################################
#############################################
begin_pcl;
int n_positions = 0;

string exp_mode = get_sdl_variable("exp_mode");

double performance; # store ratio of correct items in here
# define some variables that we are going to use later
# items holds the learning units bitmaps, the images
array <bitmap> items[0];               
# this is where the learning units (15x20 units) are saved, it defines the order of the trials
# one learning unit is a sequence of item1, item2 + 3 choices for item3
# in our case, the first learning unit is used for testing
array<string> learning_units[0][0][0]; 

# load functions and settings
include "settings_parallel.pcl";
include "functions_parallel.pcl";
trial_prompt.set_duration(0); # there is no timeout in the post-test

# if we are in testing mode, set debug participant 0
string participant_number = logfile.subject();
if participant_number=="" then 
	participant_number="0";
   debuglog("WARNING! NO PARTICIPANT NUMBER SET, USING DEFAULT 0");
	trial_warning_participant_number.present();
end;

# check if no port codes are written (ie. we are in testing mode, not recording)
string write_codes = get_sdl_variable("write_codes");
if write_codes!="true" then
	debuglog("WARNING! WRITE_CODES=FALSE");
	trial_warning_write_codes.present();
end;
n_positions = read_learning_units(participant_number)[1].count();

### START write header of file that contains the data
response_log("Trial; Seq1; Seq2; Options; Correct; Choice; RT");
## END write header


sub prepare_feedback(int selected, int correct) begin
	# reset all boxes to black
	stim_selected.set_duration(duration_show_selected*2);
	picture_selected.set_part(3, box_black);
	picture_selected.set_part(5, box_black);
	picture_selected.set_part(7, box_black);

	# set selected picture to have a frame around it
	if selected>0 then 
		picture_selected.set_part(selected*2+1, box_selected);
	end;
end;


###### MAIN TESTING BLOCK
# One block is defined as a run through all 20 starting points in the sequence
# will return the ratio of correctly named sequences tuples
sub double show_testing_block begin
	array<int> responses[0];
	loop int trial_nr = 1 until trial_nr > n_positions begin;
	   debuglog("------ trial " + string(trial_nr) + "/" + string(n_positions));
	   
		# set all items, see functions_parallel.pcl
		# take block nr 15, nobody should have seen it before.
		set_seq(15, trial_nr);
		
		trial_seq.present(); # show item1, then item1+item2
		trial_prompt.present(); # show item1, item2 + choice of 3 (item3)
		
		stimulus_data last = stimulus_manager.last_stimulus_data();
		array<int> correct_button[1]; # holds the correct button that was expected to be pressed
		stim_prompt.get_target_buttons(correct_button);
		
		# we know which button has been pressed and which should have been pressed
		prepare_feedback(last.button(), correct_button[1]);
				
		trial_selected.present();
		
		# some logging
		if last.type() == stimulus_hit then
			debuglog("correct");
			responses.add(1);
		elseif last.type() == stimulus_incorrect then
			debuglog("wrong");
			responses.add(0);
		elseif last.type() == stimulus_miss then 
			# currently I'm not using this setting at all, so should not show in log
			debuglog("too slow");
			responses.add(0);
		end;
		
		# there was an error in the response log, it logs from learning units 1, but should log from 15.
		string log_message = string(trial_nr) + "; " + strjoin(learning_units[15][trial_nr], "; ")
									 + "; " + string(last.button()) + "; " + string(last.reaction_time());
		response_log(log_message);
		
		trial_nr = trial_nr + 1;
	end;

	# calculate how many correct trials were present.
	double average_correct = arithmetic_mean(responses);
	return average_correct;
end;



#################################
###### START OF DISPLAYING ######
#################################

screen_welcome.present();

## read learning units from the csv file for that subject.
learning_units.assign(read_learning_units(participant_number));
items.assign(read_sequence(participant_number));

## Show the testing trials, starting from each item once, ask to tell which item followed
performance = show_testing_block();
debuglog("Correct: " + string(100*performance)+"%");

string string_exp_end = "Herzlichen Glückwunsch, Du hast es geschafft!\n\nDer Experimentteil ist beendet.";
trial_final.present();

print_correct_sequence();

