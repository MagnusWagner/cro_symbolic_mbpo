program_start = """
% Define the input data as facts
action(0..22).
previous_time(-5..-1).
% Define the input data as facts
action(0..22).
property(0, summercrop, 0).
property(0, drought_resistance, 0).
property(0, earliest_harvest, 13).
property(0, latest_sowing, 25).
property(0, ground_type, 0).
property(1, summercrop, 0).
property(1, drought_resistance, 1).
property(1, earliest_harvest, 13).
property(1, latest_sowing, 22).
property(1, ground_type, -1).
property(1, ground_type, 0).
property(2, summercrop, 1).
property(2, drought_resistance, 0).
property(2, earliest_harvest, 19).
property(2, latest_sowing, 9).
property(2, ground_type, 0).
property(2, ground_type, 1).
property(3, summercrop, 1).
property(3, drought_resistance, -1).
property(3, earliest_harvest, 19).
property(3, latest_sowing, 11).
property(3, ground_type, 0).
property(3, ground_type, 1).
property(4, summercrop, 1).
property(4, drought_resistance, -1).
property(4, earliest_harvest, 25).
property(4, latest_sowing, 12).
property(4, ground_type, 0).
property(5, summercrop, 1).
property(5, drought_resistance, 0).
property(5, earliest_harvest, 23).
property(5, latest_sowing, 9).
property(5, ground_type, 0).
property(6, summercrop, 0).
property(6, drought_resistance, 0).
property(6, earliest_harvest, 19).
property(6, latest_sowing, 31).
property(6, ground_type, 0).
property(6, ground_type, 1).
property(7, summercrop, 1).
property(7, drought_resistance, -1).
property(7, earliest_harvest, 20).
property(7, latest_sowing, 8).
property(7, ground_type, 0).
property(7, ground_type, 1).
property(8, summercrop, 0).
property(8, drought_resistance, 0).
property(8, earliest_harvest, 19).
property(8, latest_sowing, 28).
property(8, ground_type, 0).
property(8, ground_type, 1).
property(9, summercrop, 0).
property(9, drought_resistance, 0).
property(9, earliest_harvest, 19).
property(9, latest_sowing, 31).
property(9, ground_type, 0).
property(9, ground_type, 1).
property(10, summercrop, 0).
property(10, drought_resistance, 0).
property(10, earliest_harvest, 19).
property(10, latest_sowing, 28).
property(10, ground_type, 0).
property(10, ground_type, 1).
property(11, summercrop, 0).
property(11, drought_resistance, 1).
property(11, earliest_harvest, 22).
property(11, latest_sowing, 28).
property(11, ground_type, -1).
property(11, ground_type, 0).
property(12, summercrop, 0).
property(12, drought_resistance, 1).
property(12, earliest_harvest, 19).
property(12, latest_sowing, 28).
property(12, ground_type, -1).
property(12, ground_type, 0).
property(13, summercrop, 1).
property(13, drought_resistance, -1).
property(13, earliest_harvest, 20).
property(13, latest_sowing, 9).
property(13, ground_type, 0).
property(14, summercrop, 1).
property(14, drought_resistance, -1).
property(14, earliest_harvest, 21).
property(14, latest_sowing, 8).
property(14, ground_type, 0).
property(15, summercrop, 1).
property(15, drought_resistance, 1).
property(15, earliest_harvest, 27).
property(15, latest_sowing, 15).
property(15, ground_type, -1).
property(15, ground_type, 0).
property(16, summercrop, 1).
property(16, drought_resistance, -1).
property(16, earliest_harvest, 25).
property(16, latest_sowing, 13).
property(16, ground_type, 0).
property(16, ground_type, 1).
property(17, summercrop, 1).
property(17, drought_resistance, -1).
property(17, earliest_harvest, 24).
property(17, latest_sowing, 13).
property(17, ground_type, 0).
property(17, ground_type, 1).
property(18, summercrop, 1).
property(18, drought_resistance, 0).
property(18, earliest_harvest, 24).
property(18, latest_sowing, 10).
property(18, ground_type, 0).
property(19, summercrop, 1).
property(19, drought_resistance, -1).
property(19, earliest_harvest, 22).
property(19, latest_sowing, 14).
property(19, ground_type, -1).
property(19, ground_type, 0).
property(20, summercrop, 0).
property(20, drought_resistance, 0).
property(20, earliest_harvest, 19).
property(20, latest_sowing, 24).
property(20, ground_type, 0).
property(20, ground_type, 1).
property(21, summercrop, 1).
property(21, drought_resistance, -1).
property(21, earliest_harvest, 23).
property(21, latest_sowing, 11).
property(21, ground_type, -1).
property(21, ground_type, 0).
property(22, summercrop, 1).
property(22, drought_resistance, -1).
property(22, earliest_harvest, 24).
property(22, latest_sowing, 13).
property(22, ground_type, -1).
property(22, ground_type, 0).
cropbreak(0, single, -1).
cropbreak(0, single, -2).
cropbreak(0, single, -3).
cropbreak(0, single, -4).
cropbreak(1, single, -1).
cropbreak(1, single, -2).
cropbreak(1, single, -3).
cropbreak(1, single, -4).
cropbreak(1, single, -5).
cropbreak(2, single, -1).
cropbreak(2, single, -2).
cropbreak(2, single, -3).
cropbreak(3, single, -1).
cropbreak(3, single, -2).
cropbreak(3, single, -3).
cropbreak(3, single, -4).
cropbreak(3, single, -5).
cropbreak(4, single, -1).
cropbreak(4, single, -2).
cropbreak(4, single, -3).
cropbreak(5, single, -1).
cropbreak(5, single, -2).
cropbreak(5, single, -3).
cropbreak(5, single, -4).
cropbreak(6, single, -1).
cropbreak(6, single, -2).
cropbreak(7, single, -1).
cropbreak(7, single, -2).
cropbreak(8, single, -1).
cropbreak(8, single, -2).
cropbreak(9, single, -1).
cropbreak(9, single, -2).
cropbreak(10, single, -1).
cropbreak(10, single, -2).
cropbreak(11, single, -1).
cropbreak(12, single, -1).
cropbreak(12, single, -2).
cropbreak(13, single, -1).
cropbreak(13, single, -2).
cropbreak(14, single, -1).
cropbreak(14, single, -2).
cropbreak(14, single, -3).
cropbreak(14, single, -4).
cropbreak(15, single, -1).
cropbreak(15, single, -2).
cropbreak(16, single, -1).
cropbreak(17, single, -1).
cropbreak(18, single, -1).
cropbreak(18, single, -2).
cropbreak(18, single, -3).
cropbreak(18, single, -4).
cropbreak(19, single, -1).
cropbreak(19, single, -2).
cropbreak(19, single, -3).
cropbreak(20, single, -1).
cropbreak(20, single, -2).
cropbreak(20, single, -3).
cropbreak(21, single, -1).
cropbreak(21, single, -2).
cropbreak(21, single, -3).
cropbreak(22, single, -1).
cropbreak(22, single, -2).

mfgroup((ge;geohnemaishaferhirse;weizentriticale)).
crop_mfgroups(6, (ge;geohnemaishaferhirse;weizentriticale)).
crop_mfgroups(7, (ge;geohnemaishaferhirse;weizentriticale)).
crop_mfgroups(8, (ge;geohnemaishaferhirse;weizentriticale)).
crop_mfgroups(9, (ge;geohnemaishaferhirse)).
crop_mfgroups(10, (ge;geohnemaishaferhirse;weizentriticale)).
crop_mfgroups(11, (ge;geohnemaishaferhirse)).
crop_mfgroups(12, (ge;geohnemaishaferhirse)).
crop_mfgroups(13, (ge;geohnemaishaferhirse)).
crop_mfgroups(14, ge).
crop_mfgroups(15, ge).
crop_mfgroups(17, ge).
mf_group_block(ge, (6;7;8;9;10;11;12;13;14;15;17), -4..-1).
mf_group_block(geohnemaishaferhirse, (6;7;8;9;10;11;12;13), -3..-1).
mf_group_block(weizentriticale, (6;7;8;10), -3..-1).
mf_group_block_max_count(geohnemaishaferhirse, 2).
mf_group_block_max_count(ge, 3).
mf_group_block_max_count(weizentriticale, 1).

apgroups(0, (blatt;fl;l)).
apgroups(1, (blatt;fl;l)).
apgroups(2, (blatt;l)).
apgroups(3, (blatt;l)).
apgroups(4, (blatt;l)).
apgroups(5, (blatt;l)).
apgroups(6, (weizendinkeltriticale;weizen)).
apgroups(7, (weizendinkeltriticale;weizen)).
apgroups(8, (weizendinkeltriticale;weizen)).
apgroups(9, weizendinkeltriticale).
apgroups(10, weizendinkeltriticale).
apgroups(12, gerste).
apgroups(13, gerste).
apgroups(16, (blatt;mais)).
apgroups(17, mais).
apgroups(18, (blatt;ruebenkruziferen)).
apgroups(19, blatt).
apgroups(20, (rapssonnenblume;ruebenkruziferen)).
apgroups(21, rapssonnenblume).
ap_group_block(blatt, (0;1;2;3;4;5;16;18;19), -1).
ap_group_block(fl, (0;1), -5..-1).
ap_group_block(l, (0;1;2;3;4;5), -4..-1).
ap_group_block(weizendinkeltriticale, (6;7;8;9;10), -1).
ap_group_block(rapssonnenblume, (20;21), -2..-1).
ap_group_block(mais, (16;17), -1).
ap_group_block(ruebenkruziferen, (18;20), -3..-1).
ap_group_block(weizen, (18;20), -2..-1).
ap_group_block(gerste, (12;13), -2..-1).

% Current configuration
"""

program_end = {
    "all": """
        % Define
        % Create AP Group filters
        blocked_by_previous(A) :- action(A), previous_actions_info(X,A), cropbreak(A, single, X).
        blocked_by_ap_group(A) :- apgroups(A,APG), previous_actions_info(X,Y), ap_group_block(APG, Y, X).

        % Create MF Group filters
        mf_group_block_active(MFG, Y) :- mf_group_block(MFG, A, Y), previous_actions_info(Y,A).
        count_mf_group(MFG, C) :- C = #count {Y : mf_group_block_active(MFG, Y)}, mfgroup(MFG).
        blocked_by_mf_group(A) :- action(A), crop_mfgroups(A, MFG), count_mf_group(MFG, C), mf_group_block_max_count(MFG, MC), C+1 > MC.

        % Create filters for properties
        blocked_by_week(A) :- action(A), property(A, earliest_harvest, EH), week_info(W),  W > EH-1, property(A, summercrop, SC), SC == 0.
        allowed_by_drywet(A) :- action(A), property(A, drought_resistance, DR), drywet_info(DW),  DR > -1, DW == 0.
        allowed_by_drywet(A) :- action(A), drywet_info(DW), DW == 1.
        allowed_by_groundtype(A) :- action(A), ground_type_info(GT), property(A, ground_type, GT).

        % Filter solution candidates
        immediate_candidate(A) :- action(A), not blocked_by_previous(A), not blocked_by_ap_group(A), not blocked_by_mf_group(A), not blocked_by_week(A), allowed_by_drywet(A), allowed_by_groundtype(A).

        % Define the output predicate
        #show immediate_candidate/1.
        """,
    "only_break_rules_and_timing": """
        % Define
        % Create AP Group filters
        blocked_by_previous(A) :- action(A), previous_actions_info(X,A), cropbreak(A, single, X).
        blocked_by_ap_group(A) :- apgroups(A,APG), previous_actions_info(X,Y), ap_group_block(APG, Y, X).

        % Create MF Group filters
        mf_group_block_active(MFG, Y) :- mf_group_block(MFG, A, Y), previous_actions_info(Y,A).
        count_mf_group(MFG, C) :- C = #count {Y : mf_group_block_active(MFG, Y)}, mfgroup(MFG).
        blocked_by_mf_group(A) :- action(A), crop_mfgroups(A, MFG), count_mf_group(MFG, C), mf_group_block_max_count(MFG, MC), C+1 > MC.

        % Create filters for properties
        blocked_by_week(A) :- action(A), property(A, earliest_harvest, EH), week_info(W),  W > EH-1, property(A, summercrop, SC), SC == 0.

        % Filter solution candidates
        immediate_candidate(A) :- action(A), not blocked_by_previous(A), not blocked_by_ap_group(A), not blocked_by_mf_group(A), not blocked_by_week(A).

        % Define the output predicate
        #show immediate_candidate/1.
        """
}