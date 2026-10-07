# Sample transcripts of the chat question

2400 training transcripts from the 800 training reports (one of each style per report), 300 test transcripts on the trained phrasings and 300 on the held-out phrasings. The persona is "You are a pathology assistant."; the instructions:

- **brief**: "Answer briefly.", "Be brief.", "Reply briefly."; held out of training: "Keep the answer brief."
- **full sentence**: "Answer in a full sentence, with the reason.", "Give a full sentence, with the reason.", "Answer with the reason, in a full sentence."; held out of training: "Explain the reason in a full sentence."
- **agent**: "Call findings() to get the analyser's findings before you answer.", "Use findings() to fetch the findings before answering.", "Before you answer, call findings() for the analyser's findings."; held out of training: "Get the findings with findings() first, then answer."

Probe questions never asked in training: "What is the patient's name?", "Is this cancer?", "What stage is it?", "Is the margin clear?". Probe lines never seen: "Keep your answers short.", "Please explain your reasoning.", "You are a helpful assistant.".

## R010 · Urothelial carcinoma, invasive into lamina propria.

```
FINDINGS
surface urothelium: denuded
nests below basement membrane: present
atypia in nests: present
nest contours: irregular
stromal reaction: desmoplasia
muscularis propria: present
inflammation: mild
```

| question | brief | full sentence |
|---|---|---|
| What is the diagnosis? | Urothelial carcinoma, invasive into lamina propria. | The findings support the diagnosis: Urothelial carcinoma, invasive into lamina propria. |
| Is muscularis propria present? | Yes, present and not involved. | Muscularis propria present, not involved. |
| Is there invasion? | Yes. | Yes: the nests have irregular contours and a desmoplastic stromal reaction. |
| Is there carcinoma in situ? | No. | No: the surface urothelium shows no atypia. |
| What are the nest contours? | Irregular. | The nests below the basement membrane have irregular contours. |
| Is the surface urothelium normal? | It is denuded. | The surface urothelium is denuded, so it cannot be assessed. |
| Is there inflammation? | Yes, mild. | Yes: there is mild chronic inflammation in the lamina propria. |
| Can this be signed out as benign? | No. | No: Urothelial carcinoma, invasive into lamina propria. |

## R004 · Benign urothelium.

```
FINDINGS
surface urothelium: normal
nests below basement membrane: absent
atypia in nests: none
nest contours: none
stromal reaction: none
muscularis propria: absent
inflammation: marked
```

| question | brief | full sentence |
|---|---|---|
| What is the diagnosis? | Benign urothelium. | The findings support the diagnosis: Benign urothelium. |
| Is muscularis propria present? | No, not identified. | Muscularis propria not identified. |
| Is there invasion? | No. | No: there are no nests below the basement membrane. |
| Is there carcinoma in situ? | No. | No: the surface urothelium shows no atypia. |
| What are the nest contours? | There are no nests. | There are no nests below the basement membrane. |
| Is the surface urothelium normal? | Yes. | Yes: the surface urothelium is normal. |
| Is there inflammation? | Yes, marked. | Yes: there is marked chronic inflammation in the lamina propria. |
| Can this be signed out as benign? | Yes. | Yes: Benign urothelium. |

## R012 · Urothelial carcinoma in situ with foci suspicious for invasion.

```
FINDINGS
surface urothelium: atypia
nests below basement membrane: present
atypia in nests: present
nest contours: irregular
stromal reaction: none
muscularis propria: present
inflammation: mild
```

| question | brief | full sentence |
|---|---|---|
| What is the diagnosis? | Urothelial carcinoma in situ with foci suspicious for invasion. | The findings support the diagnosis: Urothelial carcinoma in situ with foci suspicious for invasion. |
| Is muscularis propria present? | Yes, present and not involved. | Muscularis propria present, not involved. |
| Is there invasion? | Suspicious, not established. | Suspicious: the atypical nests have irregular contours or a stromal reaction, but not both. |
| Is there carcinoma in situ? | Yes. | Yes: the surface urothelium shows full-thickness atypia. |
| What are the nest contours? | Irregular. | The nests below the basement membrane have irregular contours. |
| Is the surface urothelium normal? | No, atypical. | No: the surface urothelium shows atypia. |
| Is there inflammation? | Yes, mild. | Yes: there is mild chronic inflammation in the lamina propria. |
| Can this be signed out as benign? | No. | No: Urothelial carcinoma in situ with foci suspicious for invasion. |

## R003 · Urothelial carcinoma in situ.

```
FINDINGS
surface urothelium: atypia
nests below basement membrane: absent
atypia in nests: none
nest contours: none
stromal reaction: none
muscularis propria: present
inflammation: none
```

| question | brief | full sentence |
|---|---|---|
| What is the diagnosis? | Urothelial carcinoma in situ. | The findings support the diagnosis: Urothelial carcinoma in situ. |
| Is muscularis propria present? | Yes, present and not involved. | Muscularis propria present, not involved. |
| Is there invasion? | No. | No: there are no nests below the basement membrane. |
| Is there carcinoma in situ? | Yes. | Yes: the surface urothelium shows full-thickness atypia. |
| What are the nest contours? | There are no nests. | There are no nests below the basement membrane. |
| Is the surface urothelium normal? | No, atypical. | No: the surface urothelium shows atypia. |
| Is there inflammation? | No. | No: there is no inflammation. |
| Can this be signed out as benign? | No. | No: Urothelial carcinoma in situ. |

## One transcript of each style

### brief (C0001, from R001)

```
SYSTEM: You are a pathology assistant. Reply briefly.
USER: Here are the findings.
FINDINGS
surface urothelium: reactive
nests below basement membrane: present
atypia in nests: absent
nest contours: rounded
stromal reaction: none
muscularis propria: absent
inflammation: mild
Is there carcinoma in situ?
ASSISTANT: No.
```

### full sentence (C0002, from R001)

```
SYSTEM: You are a pathology assistant. Answer in a full sentence, with the reason.
USER: Here are the findings.
FINDINGS
surface urothelium: reactive
nests below basement membrane: present
atypia in nests: absent
nest contours: rounded
stromal reaction: none
muscularis propria: absent
inflammation: mild
What are the nest contours?
ASSISTANT: The nests below the basement membrane have rounded contours.
```

### agent (C0003, from R001)

```
SYSTEM: You are a pathology assistant. Call findings() to get the analyser's findings before you answer.
USER: Is there invasion?
ASSISTANT: findings()
TOOL:
FINDINGS
surface urothelium: reactive
nests below basement membrane: present
atypia in nests: absent
nest contours: rounded
stromal reaction: none
muscularis propria: absent
inflammation: mild
ASSISTANT: No.
```
