def detectHallucination( Answer, Sources ):

    CombinedText = " ".join( [ Doc.page_content for Doc in Sources ] )

    AnswerWords = Answer.split()
    MatchCount = 0

    for Word in AnswerWords:
        if Word.lower() in CombinedText.lower():
            MatchCount += 1

    Ratio = MatchCount / max( len(AnswerWords), 1 )

    if Ratio < 0.30:
        return True

    return False