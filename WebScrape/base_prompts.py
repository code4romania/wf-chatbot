        base_prompt = (
        f"Given the title: '{entry['heading']}' and the summary: '{entry['summary']}', "
        f"generate 5 short, distinct questions in {language} that someone curious about this topic might naturally ask."
        "before reading any details! "
        "IMPORTANT: The questions should be formulated such that on their own they should makes sense without the context. BAD question for example: What is the purpose of these measures? This doesnt makes sense as we dont have a context of 'these' "
        "Do not include the words “title” ,'these'or “summary” in your questions"
        "Return only a Python list literal of exactly 5 items."
        )
        
        base_prompt = (
        f"Given the title: '{entry['heading']}' and the summary: '{entry['summary']}', "
        f"generate 5 short, distinct questions in {language} that someone curious about '{entry['heading']}' might naturally ask "
        "BEFORE reading any details! "
        "IMPORTANT: Each question MUST be fully understandable on its own, as if seen in isolation. It should not require knowledge of the title or summary to grasp what the question is about. "
        "For example, if the title is 'New Community Gardening Program Launch', a BAD question is 'What are the benefits?' (Benefits of what?). "
        "Other BAD Examples that LACK CONTEXT: "
        "  - 'What is the purpose of these measures?' (Context needed: 'these measures' is vague). "
        "  - 'Is participation open to both adults and children?' (Context needed: 'Participation in what activity/program?'). "
        "  - 'How can I apply?' (Context needed: 'Apply for what?'). "
        "A GOOD question is 'What are the benefits of the new community gardening program?' or 'How can one join the new community gardening program?'. "
        "If the question is about a specific program, event, or concept mentioned in the title, try to include that specific program, event, or concept in the question itself. "
        "Do not include the words “title”, 'these', or “summary” in your questions. "
        "Return only a Python list literal of exactly 5 items."
        )
        
                base_prompt = (
            f"Given the title: '{entry['heading']}' and the summary: '{entry['summary']}', "
            f"generate 5 short, distinct questions in {language}. "
            "You are creating a database of frequently asked questions where each question must be stand-alone and self-explanatory. "
            "Therefore avoid saying This or These! "
            "Someone searching this database should understand the topic of the question just by reading the question itself, without referring to any title or summary. "
            "For example, if the topic is 'Benefits of Regular Exercise', a POOR question is 'How often should one do it?'. "
            "A BETTER question is 'How often should one engage in regular exercise?'. "
            "The questions should be things a person curious about '{entry['heading']}' might ask before knowing details. "
            "Do not include the words “title”, 'these','this' or “summary” in your questions. "
            "Return only a Python list literal of exactly 5 items."
        )