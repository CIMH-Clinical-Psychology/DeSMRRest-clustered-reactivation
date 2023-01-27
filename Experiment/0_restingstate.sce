scenario = "resting_state1";
$write_codes = true; # needs to be disabled if not using the MEG computer
$resting_state_duration_in_minutes = 8;
active_buttons = 3;
button_codes = 127, 127, 127; # we dont care which button is pressed exactly, but we want to know if a button has been pressed to remove the example
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
## 10		 | start RS session
## 11		 | end RS session
## 127    | button press has happened
## 255    | start and end of session
###############################################################


begin;

# load the sound played when resting state recording completed.

sound {wavefile{ filename="audio/completed.wav"; }; attenuation=0.15;} sound_end;

####################################
### Trial Screens ##################
####################################

trial { # Welcome screen
	picture{
		default_code = "welcome screen";
		text{caption="Willkommen zu Teil 1 des Experiments!

Bitte drücke eine Taste, um zu beginnen.";};
		x = 0;
		y = 0;
   };
	port_code = 255;
	code = "Start trigger (255)";
	delta_time = 0;
	duration = response;
	picture{
		text{caption="In diesem Teil des Experiments messen wir Dein Gehirn im wachen Ruhezustand.
Bitte versuche in den nächsten $resting_state_duration_in_minutes Minuten Deine Augen geschlossen zu halten
und an nichts Bestimmtes zu denken.

						  Bitte drücke eine Taste, um fortzufahren.   ";};
		x = 0;
		y = 0;
	};
	code = "instruction screen";
   duration = response;

	picture{
			text{caption="Bitte entspanne Dich. 
Du wirst benachrichtigt, wenn die $resting_state_duration_in_minutes Minuten vorbei sind.

Wenn Du bereit bist, schließe Deine Augen und\n drücke eine Taste, um zu beginnen.";};
		x = 0;
		y = 0;
	};
   duration = response;
	code = "Welcome screen";
} screen_welcome;


# resting state, 2x resting_state_duration_in_minutes
trial{
	stimulus_event{
		picture{text{caption="+";font_color=150,150,150;};x=0;y=0;};
		duration = '$resting_state_duration_in_minutes *60*1000';
		port_code = 10;
		code = "RS1 start (10)";
	}stim_rs;
}trial_rs;



# end of resting state, now we play a tone
trial{
	sound sound_end;
	picture{text{caption="Herzlichen Glückwunsch, Du hast es geschafft!\n\nDer Experimentteil ist beendet.\n\nDrücke eine Taste.";};x=0;y=0;};
	duration=next_picture;
	port_code = 11;
	code = "RS1 end (11)";
	
	picture{text{caption="Drücke eine Taste um zu Beenden.";};x=0;y=0;};
	time=3000;
	duration=response;
	port_code = 255;
	code="End trigger (255)";
}trial_end;
# end of resting state, now we play a tone


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
string participant_number = logfile.subject();
if participant_number=="" then 
	participant_number="0";
	trial_warning_participant_number.present();
end;

# check if no port codes are written (ie. we are in testing mode, not recording)
string write_codes = get_sdl_variable("write_codes");
if write_codes!="true" then
	trial_warning_write_codes.present();
end;

screen_welcome.present();

trial_rs.present();
trial_end.present();
