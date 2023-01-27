scenario = "localizer";
$write_codes = true; # needs to be false if not using the MEG computer
$exp_mode = "category_graph"; # which sequence mode is used: graph or category or category_graph
$language = "de";
active_buttons = 3;
button_codes = 127, 127, 127; # we dont care which button is pressed exactly, but we want to know if a button has been pressed to remove the example
response_logging = log_active;
response_matching = simple_matching;
default_font_size = 26;
default_font = "Arial";
response_port_output = true; # do not write button codes to MEG trigger port
write_codes = $write_codes; # dont change this, change the SDL var above

# In the SDL part of this file, the elements are only declared.
# All settings are stored and set in a separate file settings_localizer.pcl
# This is done to make changes more easily available in one place.
# This affects mainly timings, positions and sizes.

###############################################################
## Port Trigger Table
## Port Code | Meaning
## ---------------------------------------------------
## 0         | don't send trigger
## 1-XX      | image 1-XX is displayed
## 98		    | fixation before audio onset
## 99        | fixation cross & audio onset
## 101-1XX   | image 1-XX is displayed with distractor
## 127       | button press has happened
## 255       | start and end trigger
## other     | debugging, ignore
###############################################################

begin; # color of the circle is set in settings_localizer.pcl
line_graphic {} feedback_correct;
line_graphic {} feedback_wrong;


# the pictures
picture {bitmap {filename="images\\dummy.jpg";};x=0;y=0;# the image picture box
         line_graphic feedback_correct;x=0;y=0;on_top=true;
         line_graphic feedback_wrong;x=0;y=0;on_top=true;

} picture_item; 
			
sound {wavefile {filename="audio\\error.wav";} wavefile_sound;}sound_item;

picture {text{caption="+";font_color=175,175,175;};x=0;y=0;} picture_fixation; # the fixation cross that is shown before the item
picture {text{caption="+";font_color=175,175,175;};x=0;y=0;} default; # whenever there is a ms gap between the pictures, this will be shown

trial { # Welcome screen
	picture{
		default_code = "welcome screen";
		text{caption="Loading experiment in mode $exp_mode in language $language. \nPlease wait...";};
		x = 0;
		y = 0;
		
		
   };
   code = "Start trigger (255)";
   port_code = 255;
	time = 0;
	duration = 2000;
	
	picture{
		default_code = "welcome screen";
		text{caption="Willkommen!
		
Bitte drücke eine Taste, um zu beginnen.
		";};
		x = 0;
		y = 0;
   };
	time = 2000;

	duration = response;
	
	picture{ # Instruction part 1
		text{caption="Im Folgenden wird Dir immer ein Paar aus Wort und Bild gezeigt.
Meist werden Wort und Bild zueinander passen. Ab und zu wird jedoch ein falsches
Bild gezeigt werden, d.h. ein Bild das nicht zum Wort gehört.
Deine Aufgabe ist es, zu erkennen, wenn das Wort nicht zum darauffolgenden 
Bild gehört. Drücke dann so schnell wie möglich eine Taste.
Dies kann durchaus sehr selten vorkommen. 

Bitte stell Dir jedes Mal wenn du das Wort hörst, 
den zugehörigen Gegenstand/Bild vor. Dadurch
kannst du am schnellsten erkennen wenn das Paar nicht passt.

						  Bitte drücke eine Taste, um fortzufahren.
						  
                    ";};
		x = 0;
		y = 0;
	};
   duration = response;
   

} screen_welcome;

trial {
	picture{ # Instruction part 2
		text{caption="Zur Übung hörst und siehst Du jetzt 5 Wort-Bild-Paare, 
von denen eines nicht zueinander passend ist.
Bitte drücke so schnell wie möglich eine Taste, 
wenn Du bemerkst, dass das Wort nicht zum Bild gehört.
						
						  Bitte drücke eine Taste, um fortzufahren.
						  
                    ";};
		x = 0;
		y = 0;
	};
   duration = response;
} screen_training_instruction;



trial{ # screen that is shown after training finishes. caption is set later dynamically
	picture{text{caption="X";}text_feedback;x = 0;y = 0;};
   duration = response;
} screen_feedback;

trial{ # end of experiment
	picture{
		text{caption="Dieser Durchlauf ist fertig!\n.
		
				Bitte drücke eine Taste und melde Dich beim Versuchsleiter.
						  
                    ";};
		x = 0;
		y = 0;

	};
   duration = response;
	picture{text{caption="";};x=0;y=0;
	};
	port_code = 255;
	code = "End trigger (255)";
	duration=250;
} screen_ending;


trial{ # The screen that is shown immediately before starting

	stimulus_event{ # more instructions
		picture {text{caption="Das Experiment beginnt gleich.

Für eine gute Datenqualität ist es wichtig
während des Experiments weitgehend ruhig zu sitzen. 
Wenn möglich, versuche zu blinzeln, wenn der Bildschirm gerade kein Bild anzeigt.
Alle paar Minuten wirst Du eine kurze Verschnaufpause haben.
Bitte nutze diese Pause, um Deine Augen auszuruhen, zu schlucken, etc.
Versuche jedes Mal wenn du das Wort hörst, dir den Gegenstand dazu vorzustellen.

Drücke eine Taste, um zu beginnen.";}; x=0; y=0;};
		code = "prestart-instruction";
		response_active=false;
		duration=response;
	};

	stimulus_event{ # prepare
		picture {text{caption="Es geht los...";}; x=0; y=0;};
		code = "prefixation2";
		response_active=false;
		duration=next_picture;
	};
	
	stimulus_event{ # one more fixation cross before start. 
	# another fixation cross is actually shown after that, so it will just feel like one long fixation cross
		picture picture_fixation;
		code = "prefixation3";
		response_active=false;
		delta_time = 1500;
		duration = 1000;
	};
}trial_pre_start;

trial{ # this trial is shown every 80 images as a break

	stimulus_event{ # instruction
		picture {text{caption="Nimm eine kurze Verschnaufpause.\nBitte bewege dich trotzdem nicht allzu sehr.\n\nDrücke eine Taste, um fortzufahren.";}; x=0; y=0;};
		code = "break";
		response_active=false;
		duration=response;
	};
	
	stimulus_event{ # prepare for continue
		picture {text{caption="Es geht weiter...";};x=0;y=0;};
		code = "break_end";
		response_active=false;
		delta_time = 1500;
		duration = 1500;
	};
	
	stimulus_event{ # show another fixation cross, see trial_pre_start[3]
		picture picture_fixation;
		code = "breakfixation";
		response_active=false;
		delta_time = 1500;
		duration = 1000;
	};
}trial_break;


trial { # trial to show fixation cross. Will be re-use every item
        # it has its own trial so that all button presses can be ignored within the trial
	stimulus_event{
		picture picture_fixation;
		code = "pre_audio_fixation";
		response_active=false; # ignore all key presses in here
		duration = next_picture;
	}stim_pre_audio;
	
	stimulus_event{ # will be played at the same time as stim_fixation
		sound sound_item;
		response_active=false; # count key presses here
		duration=next_picture;
	}stim_audio;

	stimulus_event{ # will be shown at the same time as stim_audio
		picture picture_fixation;
		code = "fixation_audio";
		response_active=false; # ignore all key presses in here
		duration = next_picture;
	}stim_fixation;
}trial_fixation;

trial {# this is the trial that will be re-used to show the actual items/images
	trial_type = first_response;
	stimulus_event{
		picture picture_item;
		code = "trial x"; # will be set dynamically
		response_active=true; # count key presses here
		duration=next_picture;
	}stim_item;
}trial_item;

trial { # just show in second run
	picture{text {caption="Der folgende Teil ist eine Wiederholung des Teils davor.
Er ist komplett gleich.

Drücke eine Taste um fortzufahren.";};x=0;y=0;};
	code = "Second run screen";
	duration=response;
}trial_second_run;

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

array<bitmap> items[0];
# define some global variables that we are going to use later
bool button_expected = false;

# load functions and settings
# note: not all functions from functions.pcl are used in this script
include "settings_localizer.pcl";
include "functions_parallel.pcl";

# write header of file that contains the response data.
# this is kind of redundant with the automatic log file of presentation
# but much easier to parse and read for a human
response_log("Distractor; KeyPress; ResponseClass");

# get participant number. if none, take dummy participant 0
string participant_number = logfile.subject();
if participant_number=="" then 
	participant_number="0_1";
	debuglog("WARNING! NO PARTICIPANT NUMBER SET, USING DEFAULT 0_1");
	trial_warning_participant_number.present();
end;

# check if no port codes are written (ie. we are in testing mode, not recording)
string write_codes = get_sdl_variable("write_codes");
if write_codes!="true" then
	debuglog("WARNING! WRITE_CODES=FALSE");
	trial_warning_write_codes.present();
end;

# load the images used
array<sound> sounds[] = load_sounds();
array<bitmap> training_items[] = read_training_images();
array<bitmap> localizer_items[] = read_sequence(participant_number);
items.assign(training_items);

# load the sequence of the 240 item presentations, as well as distractor positions
# order[][] = [int image_number][bool is_distractor]
array<int> order[][] = read_localizer_order(participant_number);

# this function is called before every item presentation.
# it will set a random time of the fixation cross,
# set the corresponding image number to picture_item
# set the sound either as a distractor (mismatching) or matching word-item pair
sub set_item(int item_nr, bool is_distractor, int position_in_order) begin; # for setting sequences of the localizer		
		# set next picture item from items (the list of bitmaps)
		picture_item.set_part(1, items[item_nr]);

		# set new random fixation duration, duration_fixation_range is set in settings.pcl
		int duration_fixation_audio = random(duration_fixation_audio_range[1], duration_fixation_audio_range[2]);
		int duration_fixation_pre_audio = random(duration_fixation_pre_audio_range[1], duration_fixation_pre_audio_range[2]);
		trial_item.set_duration(duration_show_item); # actually this does not need to be done every time, just to make sure
		trial_fixation.set_duration(duration_fixation_pre_audio + duration_fixation_audio); # new random fixation cross time
		stim_audio.set_time(duration_fixation_pre_audio);
		stim_fixation.set_time(duration_fixation_pre_audio);	
		
		#term.print("Pre-Fix: " + string(duration_fixation_pre_audio) + "ms, Fix: " + string(duration_fixation_audio) + "ms\n");
		
		# now set the codes to appear in the logfile
		stim_item.set_target_button(0);	# reset target button. 0 = no target button
		
		# make distractor feedback item invisible
		feedback_correct.set_line_color(0,0,0,0); # just make invisible
		feedback_correct.redraw();
		feedback_wrong.set_line_color(0,0,0,0); # just make invisible
		feedback_wrong.redraw();
			
		if is_distractor then;
		   # as distractor get any other sound number, that is randomly far away from the current one
			int distractor_idx = item_nr;
			int prev_item = order[min(position_in_order+1, order.count())][1];
			int next_item = order[max(position_in_order-1, 1)][1];
		   loop until distractor_idx!=item_nr 
					  && distractor_idx!=prev_item 
					  && distractor_idx!=next_item begin;
				distractor_idx = random(1, items.count()); # get new random distractor
			end;
			sound distractor_sound = get_sound(items[distractor_idx].filename(), sounds);
			stim_audio.set_stimulus(distractor_sound);
			trial_item.set_type(first_response); # any button press ends the trial and displays the green circle as feedback
			stim_item.set_target_button({1, 2, 3}); # button 1,2,3 (=all) buttons are valid responses
			stim_item.set_event_code("[DISTRACTOR] " + string(item_nr) + " (" + string(100 + item_nr) +")"); # make clear that a distractor is shown
			stim_item.set_port_code(100 + item_nr); # add 100 to the MEG trigger port to indicate trial that is inkongruent
			button_expected = true; # we expect a button to be pressed
		else;
			sound curr_sound = get_sound(items[item_nr].filename(), sounds);
			stim_audio.set_stimulus(curr_sound);
			trial_item.set_type(first_response); # trial does not end if button is pressed incorrectly
			stim_item.set_event_code(string(item_nr)); # set event code (logfile) to item number
			stim_item.set_port_code(item_nr); # set port code to item number
			stim_item.set_response_active(true); # record false alarms
			button_expected = false; # no buttonpress is expected
		end;
end;

# after a button has been pressed, we check which button that was and
# classify the response accordingly. presentation more or less
# does that for us anyway, but I calculate it manually just to make sure
# the function will return TRUE if the button was pressed correctly, else FALSE
sub bool process_response begin;
	# get how many button presses happened during the last trial.
	int count = response_manager.response_count();
	# contains data for the last stimulus, ie. which button was pressed etc
	stimulus_data last = stimulus_manager.last_stimulus_data(); 
	
	# we show the feedback red/green circle at least $duration_show_feedback$ ms (=500ms)
	# however, if the image would normally stay much longer, we show it longer.
	# example: each item is shown 1500 ms, participant recognizes distractor after 500ms, 
	# so we show the green feedback circle for the remaining 1000ms.
	# however, if the participants presses the button at 1450ms, we shown the feedback nevertheless for 500 ms (duration_show_feedback)
	int t_show_feedback = max(duration_show_item - last.reaction_time(), duration_show_feedback);
	
	string type = ""; # CORRECT PRESS, MISS, FALSE ALARM or CORRECT NON-PRESS
	
	# correct detection
	if button_expected && count>0 then
		type = "CORRECT PRESS";
		debuglog("+ " + type);
		feedback_correct.set_line_color(0, 150, 0, 255); # make green
		feedback_correct.redraw();

		# we re-use the item trial and stimulus to show the feedback
		# for this to be possible, we need to disable all responsiveness and port code triggering
		trial_item.set_duration(t_show_feedback); # show for calculated minimum of 500 ms
		stim_item.set_port_code(0); # dont send port trigger to MEG
		stim_item.set_target_button(0); # dont respond to buttons
		stim_item.set_response_active(false); # again: dont record any button presses
		stim_item.set_event_code("feedback -> correct");
		trial_item.present(); # present feedback
		
	# miss
	elseif button_expected && count==0 then
		type = "MISS";
		debuglog("- " + type);
		# same as above at CORRECT PRESS, only show a red circle to indiciate failure
		feedback_wrong.set_line_color(150, 0, 0, 255); # make red
		feedback_wrong.redraw();
		trial_item.set_duration(duration_show_feedback); # trial time is up, but still display feedback for 500 ms
		stim_item.set_port_code(0);
		stim_item.set_target_button(0);
		stim_item.set_response_active(false);
		stim_item.set_event_code("feedback -> wrong");
		trial_item.present();

	# false alarm 
	elseif !button_expected && count>0 then
		# press where non was expected. record it.
		type = "FALSE ALARM";
		debuglog("- " + type);
		feedback_wrong.set_line_color(150, 0, 0, 255); # make red
		feedback_wrong.redraw();

		# we re-use the item trial and stimulus to show the feedback
		# for this to be possible, we need to disable all responsiveness and port code triggering
		trial_item.set_duration(t_show_feedback); # show for calculated minimum of 500 ms
		stim_item.set_port_code(0); # dont send port trigger to MEG
		stim_item.set_target_button(0); # dont respond to buttons
		stim_item.set_response_active(false); # again: dont record any button presses
		stim_item.set_event_code("feedback -> false alarm");
		trial_item.present(); # present feedback
		
	# correct non-response
	elseif !button_expected && count==0 then	
		# participant did nothing, just as requested.
		type = "CORRECT NON-PRESS";
		debuglog("+ " + type);
	end;
	
	# write response to log file
	
	string string_message = string(button_expected) + "; " + string(count>0) + "; " + type;
	response_log(string_message);
	return (button_expected && count>0);
end; 

# give a few examples, this is shown before the actual localizer images
# for training purposes. will only end if the participant makes no mistakes (false alarms) and detects the distractor.
# 5 word-image-pairs will be shown randomly and one of will be mismatching, which needs to be detected.
sub show_training begin;

	# we loop this until the participant has recognized all items correctly
	loop bool all_correct=false until all_correct begin;	
		int correct = 0; # record how many are recognized correctly
		int false_alarm = 0; # record how many incorrect presses there were
		# at this random position, mismatching word-image-pair will be shown
		int show_distractor_at = random(1, n_training_items);
		int item_nr = 0;
		int rnd = 0;
		loop int i=1 until i > n_training_items begin;
		
			# select one random training item from the training set
			# make sure it's not the same as the item from the previous time.
			loop until rnd!=item_nr begin
				rnd = random(1, items.count());
			end;
			item_nr = rnd;
			
			# show the circle only if the current item was drawn before
			bool is_distractor = i==show_distractor_at;

			# add remark if distractor should be shown
			string with_distractor = "";
			if is_distractor then; 
				with_distractor="[DISTRACTOR] " ;
			end;
			debuglog("TRAINING " + string(i) + "/" + string(n_training_items) + " item " + with_distractor + split_filename(items[item_nr].filename() +  " - "));
			
			# use the function defined above to set everything up
			set_item(item_nr, is_distractor, 2);
			# however, overwrite the port and event code to make sure these are training items, and not the actual localizer
			stim_item.set_event_code(with_distractor + "TRAINING " + string(item_nr));
			stim_pre_audio.set_event_code("TRAINING fixation pre audio");
			stim_pre_audio.set_port_code(0);
			stim_fixation.set_event_code("TRAINING fixation audio");			
			stim_item.set_port_code(0);

			# present fixation cross and item
			trial_fixation.present();
			trial_item.present();
			
			# do not use the process_response function as we will then record the items,
			# however, this just the training, so we do not want any of this in the log files.
			# the code is however more or less equivalent to the function process_response()
			int count = response_manager.response_count();
			stimulus_data last = stimulus_manager.last_stimulus_data();
			int t_show_feedback = max(duration_show_item - last.reaction_time(), duration_show_feedback);

			if (count>0) && is_distractor then;
				correct = correct+1;
				feedback_correct.set_line_color(0, 150, 0, alpha); # just make green
				feedback_correct.redraw();

				trial_item.set_duration(t_show_feedback);
				stim_item.set_target_button(0);
				stim_item.set_response_active(false);
				stim_item.set_event_code("feedback -> correct");
				trial_item.present();

			elseif is_distractor && count==0 then
				feedback_wrong.set_line_color(150, 0, 0, alpha); # make red
				feedback_wrong.redraw();
				trial_item.set_duration(duration_show_feedback);
				stim_item.set_target_button(0);
				stim_item.set_response_active(false);
				stim_item.set_event_code("feedback -> wrong");
				trial_item.present();

			elseif (count>0) && !is_distractor then;
				false_alarm = false_alarm+1;
				feedback_wrong.set_line_color(150, 0, 0, alpha); # just make red
				feedback_wrong.redraw();
				trial_item.set_duration(duration_show_feedback);
				stim_item.set_target_button(0);
				stim_item.set_response_active(false);
				stim_item.set_event_code("feedback -> false alarm");
				trial_item.present();
			end;
			
			
			i = i+1;
		end;
		
		
		# now we prepare the feedback for the training block
		string feedback;
		if correct>0 then
			feedback = feedback + "Du hast das nicht passende Wort-Bild-Paar entdeckt! Super.\n\n"
		else
			feedback = feedback + "Du hast das nicht passende Wort-Bild-Paar nicht entdeckt.\n\n"
		end;
		
		if false_alarm==0 then
			feedback = feedback + "Du hast kein Mal falsch gedrückt! Super."
		elseif false_alarm<(n_training_items-1) then
			feedback = feedback + "Du hast "+ string(false_alarm) +"x falsch gedrückt. \nBitte drücke nur, wenn das Wort-Bild-Paar nicht passt."
		else
			feedback = feedback + "Du hast jedes Mal eine Taste gedrückt. \nBitte drücke nur, wenn das Wort-Bild-Paar nicht passt."
		end;
		
		if correct>0 && false_alarm==0 then 
			all_correct = true;
			feedback =  "Das Training ist beendet\n\n" + feedback + "\n\nDrücke eine Taste, um das Experiment zu beginnen.";
		else;
			all_correct = false;
			feedback = feedback + "\n\nDrücke eine Taste, um das Training zu wiederholen.";
		end;
		
		# show feedback of training to the user. 
		# did he get everything correct ie. no false alarm, distractor detected?
		debuglog("Found? " + string(correct>0));
		debuglog("False Alarm? " + string(false_alarm));
		text_feedback.set_caption(feedback);
		text_feedback.redraw();
		screen_feedback.present();
	end;
	# reset fixation stimulus port and event code before the main localizer trial.
end;


# show all 16 items in total 240 times, where each transition
# from one item to another appears exactly one time.
sub show_items begin;
	loop int i=1 until i>order.count() begin;
		int item_nr = order[i][1]; # get the current image number
		bool is_distractor = bool(order[i][2]); # display mismatching audio-visual pair?
		string with_distractor = "";
		if is_distractor then; with_distractor="[DISTRACTOR] " end;
		debuglog(string(i) + "/" + string(order.count()) + " item " + with_distractor + split_filename(items[item_nr].filename()) +  " - ");
		
		# set item and show distractor or not
		set_item(item_nr, is_distractor, i);
		# present fixation cross and then item
		trial_fixation.present();
		trial_item.present();
		
		# process and display feedback if applicable
		process_response();
				
		# if there is a break indicated at this trial number, show it now
		if contains(break_at, i) then
			debuglog("BREAK");
			trial_break.present();
			debuglog("CONTINUE");
		end;

		i = i+1;
	end;
end;

debuglog("Runtime: ~" + calculate_runtime(localizer_items.count(), order.count()));

#########################
##### START OF DISPLAYING
screen_welcome.present();

# first show training with training items, but only at the first run
# if the run isn't 1, then skip this, the participant should have understood it by now
if participant_number.substring(participant_number.count(), 1)=="1" then
	screen_training_instruction.present();
   show_training();
else 
	trial_second_run.present();
end;
stim_pre_audio.set_event_code("fixation pre audio (98)");
stim_pre_audio.set_port_code(98);
stim_fixation.set_event_code("fixation audio (99)");
stim_fixation.set_port_code(99);
# now switch the to the actual items
items.assign(localizer_items);

trial_pre_start.present();
show_items();

screen_ending.present();
