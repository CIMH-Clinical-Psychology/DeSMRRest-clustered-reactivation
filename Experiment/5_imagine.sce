scenario = "imagine";
$write_codes = true; # needs to be disabled if not using the MEG computer
$exp_mode = "category_graph"; # which sequence mode is used: graph or category or category_graph
$language = "de";
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
## 0         | don't send trigger
## 1XX		 | first item of the sequence triplet, where XX is the image number
## 2		 | second item of the sequence triplet is displayed in addition
## 3		 | three choices are displayed in addition
## 4		 | the user selection is displayed (blue outline around item)
## 99		 | fixation cross before first image
## 255    | start and end trigger
###############################################################

# All settings are stored and loaded via PCL in settings.pcl
# this file is more or less a copy of 2_graph_learning.
# however, I think it is cleaner to have a separate script,
# and the main difference is that in the post-test there is no feedback given.
begin;
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
	delta_time = 0;
	duration = 2000;
	
	
	picture{ # Instruction part 1
		text{caption="
In diesem, letzten Teil des Experiments wirst Du noch einmal
die Worte der Objekte aus der Aufgabe davor hören. 
Deine Aufgabe ist es Dir jedes Objekt das gesagt wurde
einmal kurz vor deinem inneren Auge vorzustellen (zB Zebra, Kuchen, etc). 
Halte hierzu bitte während des Experiments die Augen geschlossen.

Alle 10 Objekte und Bilder werden mehrmals vorkommen. 
Es gibt keine bestimmte Reihenfolge. Du musst dir nichts merken.
Alle 5 Sekunden wird das nächste Wort gesagt werden.

Drücke eine Taste um fortzufahren.						  
                    ";};
		x = 0;
		y = 0;
	};
   duration = response;
  	delta_time = 2000;

	picture{ # Instruction part 1
		text{caption="
Bitte schließe die Augen und drücke eine Taste um zu beginnen.

Sobald du ein Wort hörst, stelle dir den Gegenstand 
aus der Aufgabe davor vor.  
                    ";};
		x = 0;
		y = 0;
	};
   duration = response;
	picture{ # Instruction part 1
		text{caption="+";};
		x = 0;
		y = 0;
	};
   duration = 3000;
} screen_welcome;



trial{
	stimulus_event{ # will be played at the same time as stim_fixation
		sound {wavefile {filename="audio\\error.wav";} wavefile_sound;}sound_item;
		response_active=false; # count key presses here
		duration=next_picture;
	}stim_audio;
		stimulus_event{ # will be shown at the same time as stim_audio
		picture {text{caption="+";font_color=175,175,175;};x=0;y=0;} picture_fixation; # the fixation cross that is shown before the item
		code = "fixation_audio";
		response_active=false; # ignore all key presses in here
		duration = 5000;
	}stim_fixation;
}trial_item;

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

trial{ # end of experiment
	picture{
		text{caption="Das Experiment ist fertig!\n
		
				Bitte drücke eine Taste und melde Dich beim Versuchsleiter.
						  
                    ";};
		x = 0;
		y = 0;

	};
	port_code = 255;
	code = "End trigger (255)";
   duration = response;
} screen_end;

###########################
#### PCL ##################
###########################
begin_pcl;
# dummy variables that are necessary to load functions_parallel
array<string> learning_units[0][0][0];
picture picture_seq1;
picture picture_seq2;
picture picture_prompt;
picture picture_selected;
picture picture_feedback;
stimulus_event stim_seq1;
stimulus_event stim_seq2;
stimulus_event stim_prompt;
int n_positions;
string exp_mode = get_sdl_variable("exp_mode");
int image_size = 100;

# define some variables that we are going to use later
# items holds the learning units bitmaps, the images
array <bitmap> items[0];               
# load functions and settings
include "functions_parallel.pcl";

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

items.assign(read_sequence(participant_number));


array<sound> sounds[] = load_sounds();

sub set_item(int item_nr) begin; # for setting sequences of the localizer		
	sound curr_sound = get_sound(items[item_nr].filename(), sounds);
	stim_audio.set_stimulus(curr_sound);
	stim_audio.set_port_code(item_nr);
	stim_audio.set_event_code("sound (" + string(item_nr)+")")
end;

#################################
###### START OF DISPLAYING ######
#################################

screen_welcome.present();
int j repetitions = 5;

loop int j=1 until j>repetitions begin;
	loop int i=1 until i>items.count() begin;
		set_item(i);
		debuglog("playing sound " + string(i) + ": " + split_filename(items[i].filename()));
		trial_item.present();
		i=i+1;
	end;
j = j+1;
end;


screen_end.present();