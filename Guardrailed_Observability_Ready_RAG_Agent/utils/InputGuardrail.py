PromptInjectionPatterns = [
    "ignore previous instructions",
    "forget your instructions",
    "system prompt",
    "reveal prompt",
    "bypass",
    "developer message"
]


def detectPromptInjection(UserQuestion):

    LowerQuestion = UserQuestion.lower()

    for Pattern in PromptInjectionPatterns:

        if Pattern in LowerQuestion:
            return True

    return False


def isOffTopic( UserQuestion, VectorStore, Threshold=1.2 ):

    Results = VectorStore.similarity_search_with_score( UserQuestion, k=1 )

    if len(Results) == 0:
        return True

    Score = Results[0][1]

    if Score > Threshold:
        return True

    return False