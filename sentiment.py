# sentiment.py
import pandas as pd
from collections import defaultdict
import re

def parse_chat(file_path):
    dates, textp, names, times, hour, minute = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    words, dicp = [], defaultdict(int)
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
                sec = sec_am_pm[0].strip()
                am_pm = sec_am_pm[1].strip()

                h = f"{h} {am_pm}"

                dates[i] = date
                textp[i] = message
                names[i] = name
                times[i] = f"{h}:{min_part}"
                hour[i] = h
                minute[i] = min_part
                i += 1

            except (ValueError, IndexError) as e:
                print(f"Skipping malformed line in parse_chat: {line}")
                continue

    for text in textp:
        for word in textp[text].split():
            word_lower = word.lower()
            dicp[word_lower] += 1
            if word_lower not in words:
                words.append(word_lower)

    df = pd.DataFrame({
        'date': list(dates.values()),
        'time': list(times.values()),
        'hour': list(hour.values()),
        'minute': list(minute.values()),
        'name': list(names.values()),
        'text': list(textp.values())
    })
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')
    return df, dicp, words

def get_word_frequency(dicp):
    return {w: dicp[w] for w in sorted(dicp, key=dicp.get, reverse=True) if w not in ["<media", "omitted>"]}

def get_user_threads(names):
    threadsp = defaultdict(int)
    for i in names:
        threadsp[names[i]] += 1
    return threadsp

def get_date_frequency(dates):
    fdatep = defaultdict(int)
    for i in dates:
        fdatep[dates[i]] += 1
    return fdatep

def get_hourly_activity(hour):
    hours = defaultdict(int)
    for h in hour:
        hours[hour[h]] += 1
    return hours

def simple_sentiment(text, dicp, dicn):
    c = 0
    for word in text.split():
        p = dicp.get(word.lower(), 0)
        q = dicn.get(word.lower(), 0)
        if p > q:
            c += 1
        elif p < q:
            c -= 1
    return "Positive" if c >= 0 else "Negative"