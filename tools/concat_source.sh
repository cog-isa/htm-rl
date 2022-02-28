destination='concatenated_sources.txt'

rm -r $destination

find ./htm_rl -type f -name '*.py' | 
    while read fname ; do
        if [[ $fname == *'__init__'* ]]; then
            echo '==' $fname '===>' Excluded!
        elif [[ $fname == *'agents/htm/'* ]]; then
            echo $fname
            echo '' >> $destination
            echo '' >> $destination
            echo "========= $fname source file listing =========" >> $destination
            echo '' >> $destination
            cat $fname >> $destination
        elif [[ $fname == *'/modules/'* ]]; then
            echo $fname
            echo '' >> $destination
            echo '' >> $destination
            echo "========= $fname source file listing =========" >> $destination
            echo '' >> $destination
            cat $fname >> $destination
        else
            echo '==' $fname '===>' Excluded!
            # echo $fname
            # echo '' >> $destination
            # echo '' >> $destination
            # echo "========= $fname source file listing =========" >> $destination
            # echo '' >> $destination
            # cat $fname >> $destination
        fi
    done

# find ./watcher -type f -name '*.py' | 
#     while read fname ; do
#         if [[ $fname == *'__init__'* ]]; then
#             echo '==' $fname '===>' Excluded!
#         else
#             echo $fname
#             echo '' >> $destination
#             echo '' >> $destination
#             echo "========= $fname source file listing =========" >> $destination
#             echo '' >> $destination
#             cat $fname >> $destination
#         fi
#     done

# find ./htm_rl -type f -name '*.yml' | 
#     while read fname ; do
#         echo $fname
#         echo '' >> $destination
#         echo '' >> $destination
#         echo "========= $fname config file listing =========" >> $destination
#         echo '' >> $destination
#         cat $fname >> $destination
#     done

# find ./htm_rl -type f -name '*.yaml' | 
#     while read fname ; do
#         echo $fname
#         echo '' >> $destination
#         echo '' >> $destination
#         echo "========= $fname config file listing =========" >> $destination
#         echo '' >> $destination
#         cat $fname >> $destination
#     done

# find ./htm_rl -type f -name '*.json' | 
#     while read fname ; do
#         echo $fname
#         echo '' >> $destination
#         echo '' >> $destination
#         echo "========= $fname config file listing =========" >> $destination
#         echo '' >> $destination
#         cat $fname >> $destination
#     done

# find ./htm_rl -type f -name '*.map' | 
#     while read fname ; do
#         echo $fname
#         echo '' >> $destination
#         echo '' >> $destination
#         echo "========= $fname environment map file listing =========" >> $destination
#         echo '' >> $destination
#         cat $fname >> $destination
#     done
