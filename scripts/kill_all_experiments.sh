#!/bin/bash
ind_start=0
echo "Killing screen sessions matching '.*experiment_*' with index >= $ind_start..."
screen -ls | grep "experiment_" | awk '{print $1}' | while read session; do
    # Extract the index (i) from the session name
    session_index=$(echo $session | sed 's/.*experiment_//')
    echo "Session index: $session_index"

    if [ "$session_index" -ge "$ind_start" ]; then
        screen -S $session -X quit
        echo "Quitting $session"
    fi
done
echo "Done!"
echo ""
echo "Current running screen sessions:"
screen -ls