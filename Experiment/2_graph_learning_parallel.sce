scenario = "sequence_learning_parallel";
$write_codes = true; # needs to be disabled if not using the MEG computer
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
## 32	  | the prompt
## 64     | the feedback
## 99	  | fixation cross before first image
## 255    | start and end trigger
###############################################################


# All settings are stored and loaded via PCL in settings.pcl

begin;

# define elements that we use later
# sizes are not defined here but in settings.pcl
# boxes are behind the picture when selected, ie green for correct, red for incorrect
# blue for selected
box {color=50,0,0,200; height=1; width=1; } box_red; # dark red with 200 alpha value
box {color=0,255,0; height=1; width=1;} box_green;
box {color=0,200,255; height=1; width=1;} box_selected;
box {color=0,0,0; height=1; width=1;} box_black;
text{caption="Richtig"; font_color=0,255,0;} text_correct;
text{caption="Falsch";  font_color=255,0,0;} text_wrong;
text{caption="Bitte schneller antworten.";  font_color=255,0,0;} text_too_slow;
text{caption="Die nächste Übungseinheit folgt gleich...";}text_training_start;
text{caption="+";font_color=150,150,150;}text_trial_start;
text{caption="Ende dieses Blocks. Du hattest X% richtig!\nBitte nimm eine kurze Pause.\n\nDrücke eine beliebige Taste, um fortzufahren.";}text_end_of_trial;
# default picture that will be shown when there is a millisecond gap between trials. should not happen.
picture {background_color =0,0,0; default_code = "default_background"; } default;

picture{# the picture with the first item of the 3-part sequence
        # size, position and content are set dynamically
	default_code = "seq1";
	bitmap{filename="images\\dummy.jpg";}; # item1
	x = 0;	y = 0;
} picture_seq1;


picture{# the second item of the 3-part excerpt
        # size, position and content are set dynamically
	default_code = "seq2";
	bitmap{filename="images\\dummy.jpg";}; # item1
	x = 0;	y = 0;
	bitmap{filename="images\\dummy.jpg";}; # item2
	x = 0;	y = 0;
} picture_seq2;


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

	# these should only be displayed during the training
	text {caption = "Welches Objekt folgt jetzt?\n Wähle 1, 2, 3";};	x = 0;	y = -250;
	text {caption="1";};x=0;y=0;
	text {caption="2";};x=0;y=0;
	text {caption="3";};x=0;y=0;
	
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


picture{ # will be shown shortly to give feedback
         # basically the same as picture_prompt but with a green box behind correct
		 # all values are set dynamically
	default_code = "feedback";
	
	bitmap{filename="images\\dummy.jpg";}; x = 100;	y = 0; #item1
	bitmap{filename="images\\dummy.jpg";}; x = 0;	y = 100; #item2
	
   box { height = 1; width = 1;}; x = 0; y = 0;	on_top = false; # poss. green border
	bitmap {filename="images\\dummy.jpg";};	x = 0;	y = 0; #item3
	
   box { height = 1; width = 1;}; x = 0; y = 0;	on_top = false; # poss. green border
	bitmap{filename="images\\dummy.jpg";};	x = 0;	y = 0; #item3
	
   box { height = 1; width = 1;}; x = 0; y = 0;	on_top = false; # poss. green border
	bitmap{filename="images\\dummy.jpg";};	x = 0;	y = 0; #item3
	
	text{caption="";};	x = 0;	y = -250;
} picture_feedback;


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
		text{caption="Willkommen zu Teil 3 des Experiments!
		
				Bitte drücke eine Taste um zu beginnen.";};
		x = 0;
		y = 0;
   };

	time = 2000;
	duration = response;
	picture{
		text{caption="Im Folgenden werden Dir verschiedene Objekte gezeigt.";}text_number_of_items1;
		x = 0;
		y = 200;
		text{caption="Deine Aufgabe ist es, die Reihenfolge dieser Objekte so gut wie möglich zu lernen.
Die Abfolge wiederholt sich (d.h. fängt am Ende wieder am Anfang an). 
Die Abfolge ändert sich während des Experiments nicht.
Es können Objekte doppelt vorkommen, d.h. es kann sowohl die Sequenz
'Haus - Boot - Fahrrad' vorkommen, als auch 'Haus - Hund - Glas'.

						  Bitte drücke eine Taste, um fortzufahren.   ";};
		x = 0;
		y = 0;
	};
		
   duration = response;

	picture{
		bitmap {filename="images\\example_sequence.png";
					scale_factor = 0.7;}; x=-250; y=200;
		bitmap {filename="images\\example_selection1.png";
					scale_factor = 0.7;}; x=250; y=250;
		bitmap {filename="images\\example_selection2.png";
					scale_factor = 0.7;}; x=250; y=135;		
		text{caption="Beispiel-Sequenz";};x = -250;y = 265;
		text{caption="Beispiel-Auschnitte";};x = 250;y = 325;
		text{caption="...";};x = 250;y = 65;
		
		text{caption="		
Du bekommst immer nur einen Ausschnitt aus zwei Objekten aus der Sequenz 
zu sehen und musst das dritte raten. Diese Ausschnitte sind zufällig gewählt, 
die Reihenfolge der Ausschnitte entspricht also nicht ihrer Position in der Sequenz. 
Es können Objekte doppelt vorkommen, und die Reihenfolge 
fängt am Ende wieder am Anfang an

Oben siehst du ein Beispiel.

						  Bitte drücke eine Taste, um fortzufahren.   ";};
		x = 0;
		y = -100;
	};
   duration = response;
	picture{
			text{caption="In der folgenden Aufgabe werden Dir immer zwei der 16 Objekte gezeigt,";}text_number_of_items2; 		
			x = 0; y = 160;
		text{caption="und Du musst bestimmen, welches Objekt als nächstes kommt.
Zuerst wirst Du die Abfolge nicht kennen und raten müssen.
Durch Ausprobieren wirst Du die Abfolge jedoch lernen.

Dein Ziel ist es, innerhalb von 6 Durchgängen mindestens 80% der Abfolge zu lernen.

						  Bitte drücke eine Taste, um fortzufahren.   ";};
		x = 0;
		y = 0;
	};
   duration = response;
	picture{
		text{caption="Zur Übung bekommst Du jetzt ein Beispiel gezeigt.
Die jetzt gezeigten Objekte sind NICHT Teil der späteren, richtigen Abfolge.
						
						  Bitte drücke eine Taste, um fortzufahren.
						  
                    ";};
		x = 0;
		y = 0;
	};
   duration = response;
} screen_welcome;

trial { # screen is shown after training is done and before main trial starts
	picture{
		text{caption="Die Übung ist nun beendet.\nGleich beginnt das Experiment.
Bitte bleibe während des Experiments so ruhig wie möglich sitzen.\n
						  Bitte drücke eine Taste, um fortzufahren.";}; x = 0; y = 0;
	};
   duration = response;

	picture{
		text{caption="Bitte lerne die Abfolge der folgenden Bilder so gut wie möglich.\n
						  Bitte drücke eine Taste, um zu beginnen.";}; x = 0; y = 0;
	};
   duration = response;

} screen_training_end;

trial { # ask participant if he wants more training examples
	picture{
		text{caption="Willst Du eine weitere Übungseinheit erhalten?

Taste 1 (oben) - Weitere Übungseinheit zeigen                      
Taste 2 (mitte) - Ich habe das Prinzip verstanden: Weiter zum Experiment";text_align=align_left;
	}; x =0; y=0;
	};
	target_button = 2;
	duration = response;
	code="continue training?";
}screen_continue_training;

## main trial
# this trial defines one sequence of 2 items + a choice of 3 following
trial { # img1, img2 and choice of 3 other imgs
	stimulus_event{picture{text text_trial_start;x = 0;y = 0;} picture_training_pre; code = "fixation (99)"; port_code=99;duration=next_picture;
		}stim_preq;
	stimulus_event{picture picture_seq1;code = "seq1 (1)"; port_code=1;	duration=next_picture; # item1
		}stim_seq1;
	stimulus_event{ picture picture_seq2;code = "seq2 (2)"; port_code=2; duration=next_picture; # item2
		}stim_seq2;
} trial_seq;

trial{ # present three options here
	trial_duration=1;
	trial_type=first_response;
	stimulus_event{picture picture_prompt;code = "choice (3)"; port_code=3; duration = next_picture;
		}stim_prompt; # item1, item2 + 3 choices
} trial_prompt;

trial { # highlight the selection and show the correct item in red or green
        # see picture_selected and picture_feedback for more information
	stimulus_event{picture picture_selected;  code="selection (4)"; port_code=32;}stim_selected;
	stimulus_event{picture picture_feedback; code="feedback (5)"; port_code=64;}stim_feedback;
}trial_feedback;

trial{ # fertig!
	picture{text{caption="Ende des Experimentteils.\nBitte drücke ein Taste und melde dich beim Experimentleiter.";}text_final;x=0;y=0;};
	duration=response;
	picture{text{caption="";};x=0;y=0;
	};
	port_code = 255;
	code = "End trigger (255)";
	duration=250;
}trial_final;

trial{ # after each learning block of 20 learning units, display trial_post_block
	picture{text text_end_of_trial;x=0;y=0;};
	duration=response;
	code = "End of block";
}trial_post_block;

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
string exp_mode = get_sdl_variable("exp_mode");
# define some variables that we are going to use later
# if we are in testing mode, set debug participant 0

# these are the number of positions in the learning graph // sequence
int n_positions = 0;
double performance; # store recent performance in this variable
# items can either hold the training or the learning bitmaps
# it will first hold the training_units, and then will be 
# reassigned to hold the learning units
array <bitmap> items[0];               
# this is where the learning units (15x20 units) are saved, it defines the order of the trials
# one learning unit is a sequence of item1, item2 + 3 choices for item3
array<string> learning_units[0][0][0]; 
array<bitmap> training_items[0];
# load functions and settings

include "settings_parallel.pcl";
include "functions_parallel.pcl";
#load training images
training_items.assign(read_training_images());

string participant_number = logfile.subject();
if participant_number=="" then 
	participant_number="0";
   debuglog("WARNING! NO PARTICIPANT NUMBER SET, USING DEFAULT 0");
	trial_warning_participant_number.present();
end;
n_positions = read_learning_units(participant_number)[1].count();

# check if no port codes are written (ie. we are in testing mode, not recording)
string write_codes = get_sdl_variable("write_codes");
if write_codes!="true" then
	debuglog("WARNING! WRITE_CODES=FALSE");
	trial_warning_write_codes.present();
end;

### START write header of file that contains the data
response_log("Block; Trial; Seq1; Seq2; Options; Correct; Choice; RT");
## END write header

array<string> main_learning_units[][][] = read_learning_units(participant_number);
array<bitmap> main_items[] = read_sequence(participant_number);

string n_items = string(main_items.count());
text_number_of_items1.set_caption("Im Folgenden werden Dir " + n_items + " verschiedene Objekte gezeigt.");text_number_of_items1.redraw();
text_number_of_items2.set_caption("In der folgenden Aufgabe werden Dir immer zwei der " + n_items + " Objekte gezeigt,");text_number_of_items2.redraw();

sub prepare_feedback(int selected, int correct) begin
	# in this function: make outline of correct green
	# and dim all others with a dark red

	# if anything has been selected, put an outline around it
	if selected>0 then 
		picture_selected.set_part(selected*2+1, box_selected);
	end;

	# in the feedback, set a green box around the correct item
	# reminder picture_feedback picture part numbering:
    # 1 = seq1, 2 = seq2, 3 = frame of choice1, 4 = choice1, 5 = frame of choice2, 6 = choice2, 7 = frame of choice3, 8 = choice 3 9 = text below
	# that means we need to set all frames, ie index 3, 5, 7. By default the box is behind the image, so it will be a frame
	# however, if we put it on top, it will overlay the image. If an alpha value is set (as for box_red), it will result in a red shading of the image.
	loop int i=1 until i>3 begin
		if i==correct then # make correct one have green outline
			picture_feedback.set_part(i*2+1, box_green);
		else # give wrong choices a red overlay
			picture_feedback.set_part(i*2+1, box_red);
			picture_feedback.set_part_on_top(i*2+1, true); # on top means it will be a red overlay over the incorrect image
		end;
		i = i+1;
	end;
	
	if selected==0 then 	# if no button has been pressed, show "too slow"
		picture_feedback.set_part(9, text_too_slow);
	elseif selected==correct then # if correct: show "richtig", remark: this will only be displayed during training
		picture_feedback.set_part(9, text_correct);
	else # if incorrect: show "falsch", remark: this will only be displayed during training
		picture_feedback.set_part(9, text_wrong);
	end;

end;

##### TRAINING BLOCK
# show training by calling this function
sub show_training_block begin
	debuglog("##### training #####");
	# set training text to be displayed
	picture_training_pre.set_part(1, text_training_start);
	# show learning units as defined in settings
	loop int trial_nr = 1 until trial_nr > n_positions begin;
	   debuglog("------ training trial " + string(trial_nr));
	    
		# set all items, see below
		set_seq(1, trial_nr);
	
		trial_seq.present(); # show item1, then item1+item2
		trial_prompt.present(); # show item1, item2 + choice of 3 (item3)
		
		stimulus_data last = stimulus_manager.last_stimulus_data();
		array<int> correct_button[1]; # holds the correct button that was expected to be pressed
		stim_prompt.get_target_buttons(correct_button);

		# we know which button has been pressed and which should have been pressed
		# according to this, prepare the feedback (green outline, display "correct" or "wrong")
		prepare_feedback(last.button(), correct_button[1]);
		
		# if the participant was slower than $max_reaction_time$, let him know
		if (last.reaction_time()>max_reaction_time) then
			text_trial_start.set_caption("Bitte schneller antworten.\n\nDie nächste Übungseinheit folgt gleich...");
			text_trial_start.redraw();
		else
			text_trial_start.set_caption("Die nächste Übungseinheit folgt gleich...");
			text_trial_start.redraw();
		end;
		
		# now display the feedback, with the correct item shown with a green outline
		trial_feedback.present();
		
		# some logging
		if last.type() == stimulus_hit then
			debuglog("correct");
		elseif last.type() == stimulus_incorrect then
			debuglog("wrong");
		elseif last.type() == stimulus_miss then 		
			# currently I'm not using this setting at all, so should not show in log
			debuglog("too slow");
		end;
		
		# after 3 trials, we check whether the participant had enough
		if trial_nr%3==0 && trial_nr>1  then
			screen_continue_training.present();
			last = stimulus_manager.last_stimulus_data();
			debuglog("Enough training? -> Button " + string(last.button()));
			if last.type()==stimulus_hit then
				break;
			end;
		end;
		trial_nr = trial_nr + 1;
		
	end;
	debuglog("Training end.\n#######");
	# set text back to learning text not training introduction text
	picture_training_pre.set_part(1, text_trial_start);
end;

###### MAIN BLOCK
# One block is defined as a run through all 20 starting points in the sequence
# will return the ratio of correctly named sequences
sub double show_trial_block(int block_nr) begin
	array<int> responses[0]; # count of correct answers. it's actually a boolean array.
	loop int trial_nr = 1 until trial_nr > n_positions begin; # run through all training units
	    debuglog("------ trial " + string(trial_nr) + "/" + string(n_positions));
		
		# set all images to their correct position
		set_seq(block_nr, trial_nr);
		
		trial_seq.present();
		trial_prompt.present();
		
		# check which button has been pressed
		stimulus_data last = stimulus_manager.last_stimulus_data();
		array<int> correct_button[1]; # retrieve the correct button
		stim_prompt.get_target_buttons(correct_button);
		prepare_feedback(last.button(), correct_button[1]);
		trial_feedback.present(); # present feedback, i.e. green outline for correct item
		
		# record whether the item was chosen correctly
		if last.type() == stimulus_hit then
			debuglog("correct");
			responses.add(1);
		elseif last.type() == stimulus_incorrect then
			debuglog("wrong");
			responses.add(0);
		elseif last.type() == stimulus_miss then 
		   # currently I'm not using this setting at all, should not be invoked
			debuglog("too slow");
			responses.add(0);
		end;
		
		string log_message = string(block_nr) + "; " + string(trial_nr) + "; " + strjoin(learning_units[block_nr][trial_nr], "; ")
									 + string(last.button()) + "; " + string(last.reaction_time());
		response_log(log_message);

		trial_nr = trial_nr + 1;
	end;
	
	# calculate how many correct trials were present.
	double average_correct = arithmetic_mean(responses); #0.0-1.0 in steps of 0.05
	return average_correct; # return the performance of this block of 20 learning units
end;



#################################
###### START OF DISPLAYING ######
#################################
screen_welcome.present();

# we push the training items into the queue.
# later we will replace that with the real learning units.
learning_units.assign(training_units);
items.assign(training_items);

## training sequence
show_training_block();

screen_training_end.present();

# remove the button descriptions "1,2,3" and the "bitte wählen" text which was used during training
loop int i=1 until i>4 begin;
	picture_prompt.remove_part(picture_prompt.part_count()-1);
	i = i+1;
end;
##### learning start
# now we switch from training to learning, so change some instructions
text_trial_start.set_caption("+"); # now it's just fixation
text_trial_start.redraw();
text_correct.set_caption(""); # don't show "richtig" anymore
text_correct.redraw();
text_wrong.set_caption(""); # don't show "falsch" anymore
text_wrong.redraw();
# reset arrays
items.resize(0); 
learning_units.resize(0);
# load the items for learning and putting them in the items array
# read learning units from the csv file for that subject.
# load training images and learning unit sequences
learning_units.assign(main_learning_units);
items.assign(main_items);


# loop through blocks of trials. 
# Each block contains 20 learning units starting at each sequence point once.
loop int block_nr = 1 until block_nr > 6 begin;
	debuglog("##### block " + string(block_nr) + " #####");
	# show one block of 20 learning units
	performance = show_trial_block(block_nr);
	debuglog("Correct: " + string(100*performance)+"%");
	# if performance is below 0.8 or just one block was done, repeat learning
	if (performance<0.8) || (block_nr==1) then 
		string string_block_end = "Ende von Block I%.\nDu hattest X% richtig.\nBitte nimm eine kurze Pause, bleib aber trotzdem ruhig sitzen.\n\nDrücke eine beliebige Taste, um fortzufahren.";
		string_block_end = string_block_end.replace("X%", string(int(performance*100)) + "%");
		string_block_end = string_block_end.replace("I%", string(block_nr));
		text_end_of_trial.set_caption(string_block_end);
		text_end_of_trial.redraw();
		trial_post_block.present();
	else # if performance was >=80%, the participant is done!
		string string_exp_end = "Herzlichen Glückwunsch, Du hast es in I% Durchgängen geschafft!\nDu hattest X% richtig.\n\nDer Experimentteil ist beendet.\n\n In knapp 10 Minuten werden wir testen, wieviel der Sequenz du noch erinnern kannst.";
		string_exp_end = string_exp_end.replace("I%", string(block_nr));
		string_exp_end = string_exp_end.replace("X%", string(int(performance*100)) + "%");
		text_final.set_caption(string_exp_end);

		text_final.redraw();
		break;
	end;
	block_nr = block_nr+1;
end;

# last but not least, print correct sequence to terminal
print_correct_sequence();
trial_final.present();
