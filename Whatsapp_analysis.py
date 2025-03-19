# Whatsapp_analysis.py
import pandas as pd
from collections import defaultdict
import re

def analyze_chat(file_path):
    dates, texts, names, times, hour, minute = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    dictWords, words, count = defaultdict(int), [], defaultdict(int)
    i = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                match = re.match(r'\[(.*?)\]', line)
                if not match:
                    continue
                timestamp = match.group(1)  # e.g., "01/11/24, 1:20:35 AM"
                rest = line[len(match.group(0)):].strip()

                if ': ' not in rest:
                    continue
                name, message = rest.split(': ', 1)

                if "end-to-end encrypted" in message.lower():
                    continue

                date, time = timestamp.split(', ', 1)
                h, m = time.split(':', 1)
                h = h.strip()  # Hour (e.g., "1")

                m_parts = m.split(':')
                if len(m_parts) < 2:
                    continue   
                min_part = m_parts[0].strip()   
                sec_am_pm = m_parts[1].strip()  

                sec_am_pm = re.split(r'[\s\u202f]+', sec_am_pm)
                if len(sec_am_pm) < 2:
                    continue  
                sec = sec_am_pm[0].strip()  # Seconds (e.g., "35")
                am_pm = sec_am_pm[1].strip()  # AM/PM (e.g., "AM")

                h = f"{h} {am_pm}"   

                dates[i] = date
                texts[i] = message
                names[i] = name
                times[i] = f"{h}:{min_part}"
                hour[i] = h
                minute[i] = min_part  # Store only the minute part
                i += 1

            except (ValueError, IndexError) as e:
                print(f"Skipping malformed line in analyze_chat: {line}")
                continue

    print("Parsed names and texts:")
    for idx in names:
        print(f"Name: {names[idx]}, Message: {texts[idx]}")

    print("Minute values:")
    for idx in minute:
        print(f"Minute[{idx}]: {minute[idx]}")


    for i in names:
        count[names[i]] += 1

    # Initialize cwords with all names
    cwords = {name: 0 for name in count.keys()}
    j = 0
    for text in texts:
        for word in texts[text].split():
            word_lower = word.lower()
            dictWords[word_lower] += 1
            if word_lower not in words:
                words.append(word_lower)
            cwords[names[j]] += 1
        j += 1

    hours = defaultdict(int)
    for h in hour:
        hours[hour[h]] += 1

    ig = 0
    for t in range(i-1):
        if hour[t] == hour[t+1]:
            if minute[t] == minute[t+1] or int(minute[t]) == int(minute[t+1]) - 1:
                continue
            elif int(minute[t+1]) - int(minute[t]) > 2:
                ig += 1
        else:
            if int(minute[t]) >= 59 and int(minute[t+1]) <= 1:
                ig += 1
            else:
                ig += 1

    interest = "Interested" if ig < (sum(count.values()) / 5) else "Not Interested"
    user_interest = "Me less interested" if cwords[list(cwords.keys())[1]] > 2 * cwords[list(cwords.keys())[0]] else "Other less interested" if cwords[list(cwords.keys())[0]] > 2 * cwords[list(cwords.keys())[1]] else "Balanced"

    return {
        'total_messages': i,
        'user_word_counts': cwords,
        'hourly_activity': hours,
        'interest': interest,
        'user_interest': user_interest,
        'df': pd.DataFrame({
            'date': list(dates.values()),
            'time': list(times.values()),
            'hour': list(hour.values()),
            'minute': list(minute.values()),
            'name': list(names.values()),
            'text': list(texts.values())
        })
    }