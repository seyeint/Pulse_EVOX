def activityNotifications(expenditure, d):
    notifs = 0
    for x in range(d, len(expenditure)):
        previous = sorted(expenditure[x - d: x])
        median = (previous[d // 2 - 1] + previous[d // 2]) / 2 if len(previous) % 2 == 0 else previous[d // 2]

        if expenditure[x] >= 2 * median:
            notifs += 1

    return notifs


activityNotifications([1,4,3,5,2], 3)