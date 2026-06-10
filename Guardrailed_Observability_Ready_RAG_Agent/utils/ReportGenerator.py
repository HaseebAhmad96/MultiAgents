from datetime import datetime

SessionLogs = []

def addSessionRecord( Question, Decision, Answer, Sources ):

    SessionLogs.append( { "Question": Question, "Decision": Decision, "Answer": Answer, "Sources": Sources } )


def saveSessionReport():

    TimeStamp = datetime.now().strftime( "%Y%m%d_%H%M%S" )

    FileName = ( f"reports/Session_{TimeStamp}.txt" )

    with open( FileName, "w", encoding="utf-8" ) as File:

        for Record in SessionLogs:

            File.write( f"Question: {Record['Question']}\n" )

            File.write( f"Decision: {Record['Decision']}\n" )
            
            File.write( f"Answer: {Record['Answer']}\n" )

            File.write( "Sources:\n" )

            for Source in Record["Sources"]:
                File.write( f"- {Source}\n" )

            File.write("\n\n" )

    return FileName