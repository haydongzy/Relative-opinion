#!/usr/bin/env python3

"""
Created on Fri Jan 13 10:37:55 2017

@author: Zhaoya Gong, Scott Hale

fields of tweets:
quoted_status_id
in_reply_to_user_id
retweeted_status

Updated: 2017-05-28
Replace newlines and tabs with space.
Separate output with tabs

Modified: 2017-10-17
filter english tweets
search keywords
retweet user
reply to user
quote user
mention user

"""

import sys
import json
import datetime
import io
import re

RE_NEWLINE_TAB=re.compile(r"[\r\n\t]",re.MULTILINE)

def read_input(file):
	for line in file:
		yield line



def mapper(line):
	line_object = json.loads(line)

	try:
		lang = line_object["lang"]
	except KeyError:
		lang = "Null"

	if lang == "en":
		try:
			if line_object["place"]["country_code"] == "US":
				country_code = RE_NEWLINE_TAB.sub(" ",line_object["place"]["country_code"])
				place_full_name = RE_NEWLINE_TAB.sub(" ",line_object["place"]["full_name"])
				place_name = RE_NEWLINE_TAB.sub(" ",line_object["place"]["name"])
				place_type = RE_NEWLINE_TAB.sub(" ",line_object["place"]["place_type"])
				place_bounding_box = line_object["place"]["bounding_box"]["coordinates"][0]
			else:
				country_code = "Not"
				place_full_name = RE_NEWLINE_TAB.sub(" ",line_object["place"]["full_name"])
				place_name = "Not"
				place_type = "Not"
				place_bounding_box = "Not"
		except KeyError:
			country_code = "Null"
			place_full_name = "Null"
			place_name = "Null"
			place_type = "Null"
			place_bounding_box = "Null"
		except TypeError:
			country_code = "Null"
			place_full_name = "Null"
			place_name = "Null"
			place_type = "Null"
			place_bounding_box = "Null"

		try:
			coord = line_object["coordinates"]["coordinates"]
			if coord[0] > -124.69968 and coord[0] < -66.97609 and coord[1] > 25.136662 and coord[1] < 49.365789 :
				tw_coor = coord
			else:
				tw_coor = "Not"
		except TypeError:
			tw_coor = "Null"
		except KeyError:
			tw_coor = "Null"

		if (country_code != "Null" and country_code != "Not") or (tw_coor != "Null" and tw_coor != "Not"):

# search for keywords "trump, realdonaldtrump, donaldtrump, hillary, clinton, hillaryclinton" with case insensitive
			try:
				ttext = line_object["text"].lower()
				if ("trump" in ttext) or ("realdonaldtrump" in ttext) or ("donaldtrump" in ttext) or ("hillary" in ttext) or ("clinton" in ttext) or ("hillaryclinton" in ttext):
					tt = line_object["text"]
				else:
					tt = "Null"
			except KeyError:
				tt = "Null"

			if tt != "Null":
				user_id = line_object["user"]["id"]
				tw_id = line_object["id"]
# get timestamp
				timestamp = line_object["created_at"]

# quote other user
				try:
					quoted_status_user_id = line_object["quoted_status"]["user"]["id"]
				except KeyError:
					quoted_status_user_id = "Null"

# reply to other user
				try:
					in_reply_to_user_id = line_object["in_reply_to_user_id"]
					if in_reply_to_user_id is None:
						in_reply_to_user_id = "Null"

				except KeyError:
					in_reply_to_user_id = "Null"

# retweet other user
				try:
					retweeted_user_id = line_object["retweeted_status"]["user"]["id"]
				except KeyError:
					retweeted_user_id = "Null"

# mention other users
				mentioned_user_ids = []
				try:
					users = line_object["entities"]["user_mentions"]
					for iu in users:
						mentioned_user_ids.append(iu["id"])
					if not mentioned_user_ids:
						mentioned_user_ids = "Null"
				except KeyError:
					mentioned_user_ids = "Null"

#print(tw_id, user_id, country_code, place_full_name, place_name, place_type, place_bounding_box, tw_coor, quoted_status_user_id, in_reply_to_user_id, retweeted_user_id, mentioned_user_ids, tt)
				vals=[tw_id, user_id, timestamp, country_code, place_full_name, place_name, place_type, place_bounding_box, tw_coor, quoted_status_user_id, in_reply_to_user_id, retweeted_user_id, mentioned_user_ids, tt]
				vals=[str(val) for val in vals]
				print("\t".join(vals))


if __name__ == "__main__":
	data=read_input(io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8'))
	for line in data:
		try:
			mapper(line)
		except Exception as e:
			sys.stderr.write("Failed to parse line with error {}\nLine was {}\n".format(e,line))

