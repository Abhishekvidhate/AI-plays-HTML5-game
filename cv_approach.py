import cv2
import numpy as np
import win32api, win32gui, win32ui, win32con, win32com.client
from PIL import Image, ImageFont, ImageDraw, ImageOps


# create training model based on the given TTF font file
# http://projectproto.blogspot.com/2014/07/opencv-python-digit-recognition.html
def createDigitsModel(fontfile, digitheight):
    font = ImageFont.truetype(fontfile, digitheight)
    samples = np.empty((0, digitheight * (digitheight / 2)))
    responses = []
    for n in range(10):
        pil_im = Image.new("RGB", (digitheight, digitheight * 2))
        ImageDraw.Draw(pil_im).text((0, 0), str(n), font=font)
        pil_im = pil_im.crop(pil_im.getbbox())
        pil_im = ImageOps.invert(pil_im)
        # pil_im.save(str(n) + ".png")

        # convert to cv image
        cv_image = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGBA2BGRA)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

        roi = cv2.resize(thresh, (digitheight, digitheight / 2))
        responses.append(n)
        sample = roi.reshape((1, digitheight * (digitheight / 2)))
        samples = np.append(samples, sample, 0)

    samples = np.array(samples, np.float32)
    responses = np.array(responses, np.float32)

    model = cv2.KNearest()
    model.train(samples, responses)
    return model


class Board(object):
    UP, DOWN, LEFT, RIGHT = 1, 2, 3, 4
    FONT = "font/ClearSans-Bold.ttf"

    def __init__(self, clientwindowtitle):
        self.hwnd = self.getClientWindow(clientwindowtitle)
        if not self.hwnd:
            return
        self.hwndDC = win32gui.GetWindowDC(self.hwnd)
        self.mfcDC = win32ui.CreateDCFromHandle(self.hwndDC)
        self.saveDC = self.mfcDC.CreateCompatibleDC()

        self.cl, self.ct, right, bot = win32gui.GetClientRect(self.hwnd)
        self.cw, self.ch = right - self.cl, bot - self.ct
        self.cl += win32api.GetSystemMetrics(win32con.SM_CXSIZEFRAME)
        self.ct += win32api.GetSystemMetrics(win32con.SM_CYSIZEFRAME)
        self.ct += win32api.GetSystemMetrics(win32con.SM_CYCAPTION)
        self.ch += win32api.GetSystemMetrics(win32con.SM_CYSIZEFRAME) * 2

        self.saveBitMap = win32ui.CreateBitmap()
        self.saveBitMap.CreateCompatibleBitmap(self.mfcDC, self.cw, self.ch)
        self.saveDC.SelectObject(self.saveBitMap)

        self.tiles, self.tileheight, self.contour = self.findTiles(self.getClientFrame())
        if not len(self.tiles):
            return
        self.digitheight = self.tileheight / 2
        self.digitsmodel = createDigitsModel(self.FONT, self.digitheight)

        self.update()

    def getClientWindow(self, windowtitle):
        toplist, winlist = [], []

        def enum_cb(hwnd, results):
            winlist.append((hwnd, win32gui.GetWindowText(hwnd)))

        win32gui.EnumWindows(enum_cb, toplist)
        window = [(hwnd, title) for hwnd, title in winlist if windowtitle.lower() in title.lower()]
        if not len(window):
            return 0
        return window[0][0]

    def getClientFrame(self):
        self.saveDC.BitBlt((0, 0), (self.cw, self.ch),
                           self.mfcDC, (self.cl, self.ct), win32con.SRCCOPY)

        bmpinfo = self.saveBitMap.GetInfo()
        bmpstr = self.saveBitMap.GetBitmapBits(True)

        pil_img = Image.frombuffer('RGB',
                                   (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
                                   bmpstr, 'raw', 'BGRX', 0, 1)

        array = np.array(pil_img)
        cvimage = cv2.cvtColor(array, cv2.COLOR_RGBA2BGRA)
        return cvimage

    def findTiles(self, cvframe):
        tiles, avgh = [], 0

        gray = cv2.cvtColor(cvframe, cv2.COLOR_BGRA2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, 1, 1, 11, 2)
        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        def findBoard(contours):  # get largest square
            ww, sqcnt = 10, None
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if w > ww and abs(w - h) < w / 10:
                    ww = w
                    sqcnt = cnt
            return sqcnt

        board = findBoard(contours)
        if board == None:
            print
            'board not found!'
            return tiles, avgh, board

        bx, by, bw, bh = cv2.boundingRect(board)
        # cv2.rectangle(cvframe,(bx,by),(bx+bw,by+bh),(0,255,0),2)
        # cv2.imshow('board',cvframe)
        # cv2.waitKey(0)
        # cv2.destroyWindow( 'board' )
        maxh = bh / 4
        minh = (maxh * 4) / 5
        count = 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if y > by and w > minh and w < maxh and h > minh and h < maxh:
                avgh += h
                count += 1
        if not count:
            print
            'no tile found!'
            return tiles, avgh, board

        avgh = avgh / count
        margin = (bh - avgh * 4) / 5
        for row in range(4):
            for col in range(4):
                x0 = bx + avgh * col + margin * (col + 1)
                x1 = x0 + avgh
                y0 = by + avgh * row + margin * (row + 1)
                y1 = y0 + avgh
                tiles.append([x0, y0, x1, y1])
                # cv2.rectangle(cvframe,(x0,y0),(x1,y1),(0,255,0),2)
        # cv2.imshow('tiles',cvframe)
        # cv2.waitKey(0)
        # cv2.destroyWindow( 'tiles' )
        return tiles, avgh, board

    def getTileThreshold(self, tileimage):
        gray = cv2.cvtColor(tileimage, cv2.COLOR_BGR2GRAY)
        row, col = gray.shape
        tmp = gray.copy().reshape(1, row * col)
        counts = np.bincount(tmp[0])
        sort = np.sort(counts)

        modes, freqs = [], []
        for i in range(len(sort)):
            freq = sort[-1 - i]
            if freq < 4:
                break
            mode = np.where(counts == freq)[0][0]
            modes.append(mode)
            freqs.append(freq)

        bg, fg = modes[0], modes[0]
        for i in range(len(modes)):
            fg = modes[i]
            # if abs(bg-fg)>=48:
            if abs(bg - fg) > 32 and abs(fg - 150) > 4:  # 150?!
                break
        # print bg, fg
        if bg > fg:  # needs dark background ?
            tmp = 255 - tmp
            bg, fg = 255 - bg, 255 - fg

        tmp = tmp.reshape(row, col)
        ret, thresh = cv2.threshold(tmp, (bg + fg) / 2, 255, cv2.THRESH_BINARY)
        return thresh

    def getTileNumbers(self, cvframe):
        numbers = []
        outframe = np.zeros(cvframe.shape, np.uint8)

        def guessNumber(digits):
            for i in range(1, 16):
                nn = 2 ** i
                ss = str(nn)
                dd = [int(c) for c in ss]
                if set(digits) == set(dd):
                    return nn
            return 0

        for tile in self.tiles:
            x0, y0, x1, y1 = tile
            tileimage = cvframe[y0:y1, x0:x1]
            cv2.rectangle(cvframe, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.rectangle(outframe, (x0, y0), (x1, y1), (0, 255, 0), 1)

            thresh = self.getTileThreshold(tileimage)
            contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            dh = self.digitheight
            digits = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if h > w and h > (dh * 1) / 5 and h < (dh * 6) / 5:
                    cv2.rectangle(cvframe, (x0 + x, y0 + y), (x0 + x + w, y0 + y + h), (0, 0, 255), 1)
                    roi = thresh[y:y + h, x:x + w]
                    roi = cv2.resize(roi, (dh, dh / 2))
                    roi = roi.reshape((1, dh * (dh / 2)))
                    roi = np.float32(roi)
                    retval, results, neigh_resp, dists = self.digitsmodel.find_nearest(roi, k=1)
                    digit = int((results[0][0]))
                    string = str(digit)
                    digits.append(digit)
                    cv2.putText(outframe, string, (x0 + x, y0 + y + h), 0, float(h) / 24, (0, 255, 0))

            numbers.append(guessNumber(digits))
        return numbers, outframe

    def getWindowHandle(self):
        return self.hwnd

    def getBoardContour(self):
        return self.contour

    def update(self):
        frame = self.getClientFrame()
        self.tilenumbers, outframe = self.getTileNumbers(frame)
        return self.tilenumbers, frame, outframe

    def copyTileNumbers(self):
        return self.tilenumbers[:]

    def getCell(self, tiles, x, y):
        return tiles[(y * 4) + x]

    def setCell(self, tiles, x, y, v):
        tiles[(y * 4) + x] = v
        return tiles

    def getCol(self, tiles, x):
        return [self.getCell(tiles, x, i) for i in range(4)]

    def setCol(self, tiles, x, col):
        for i in range(4):
            self.setCell(tiles, x, i, col[i])
        return tiles

    def getLine(self, tiles, y):
        return [self.getCell(tiles, i, y) for i in range(4)]

    def setLine(self, tiles, y, line):
        for i in range(4):
            self.setCell(tiles, i, y, line[i])
        return tiles

    def validMove(self, tilenumbers, direction):
        if direction == self.UP or direction == self.DOWN:
            for x in range(4):
                col = self.getCol(tilenumbers, x)
                for y in range(4):
                    if (y < 4 - 1 and col[y] == col[y + 1] and col[y] != 0):
                        return True
                    if (direction == self.DOWN and y > 0 and col[y] == 0 and col[y - 1] != 0):
                        return True
                    if (direction == self.UP and y < 4 - 1 and col[y] == 0 and col[y + 1] != 0):
                        return True
        if direction == self.LEFT or direction == self.RIGHT:
            for y in range(4):
                line = self.getLine(tilenumbers, y)
                for x in range(4):
                    if (x < 4 - 1 and line[x] == line[x + 1] and line[x] != 0):
                        return True
                    if (direction == self.RIGHT and x > 0 and line[x] == 0 and line[x - 1] != 0):
                        return True
                    if (direction == self.LEFT and x < 4 - 1 and line[x] == 0 and line[x + 1] != 0):
                        return True
        return False

    def moveTileNumbers(self, tilenumbers, direction):
        def collapseline(line, direction):
            if (direction == self.LEFT or direction == self.UP):
                inc = 1
                rg = xrange(0, 4 - 1, inc)
            else:
                inc = -1
                rg = xrange(4 - 1, 0, inc)
            pts = 0
            for i in rg:
                if line[i] == 0:
                    continue
                if line[i] == line[i + inc]:
                    v = line[i] * 2
                    line[i] = v
                    line[i + inc] = 0
                    pts += v
            return line, pts

        def moveline(line, directsion):
            nl = [c for c in line if c != 0]
            if directsion == self.UP or directsion == self.LEFT:
                return nl + [0] * (4 - len(nl))
            return [0] * (4 - len(nl)) + nl

        score = 0
        if direction == self.LEFT or direction == self.RIGHT:
            for i in range(4):
                origin = self.getLine(tilenumbers, i)
                line = moveline(origin, direction)
                collapsed, pts = collapseline(line, direction)
                new = moveline(collapsed, direction)
                tilenumbers = self.setLine(tilenumbers, i, new)
                score += pts
        elif direction == self.UP or direction == self.DOWN:
            for i in range(4):
                origin = self.getCol(tilenumbers, i)
                line = moveline(origin, direction)
                collapsed, pts = collapseline(line, direction)
                new = moveline(collapsed, direction)
                tilenumbers = self.setCol(tilenumbers, i, new)
                score += pts

        return score, tilenumbers

    # AI based on "term2048-AI"


# https://github.com/Nicola17/term2048-AI
class AI(object):
    def __init__(self, board):
        self.board = board

    def nextMove(self):
        tilenumbers = self.board.copyTileNumbers()
        m, s = self.nextMoveRecur(tilenumbers[:], 3, 3)
        return m

    def nextMoveRecur(self, tilenumbers, depth, maxDepth, base=0.9):
        bestMove, bestScore = 0, -1
        for m in range(1, 5):
            if (self.board.validMove(tilenumbers, m)):
                score, newtiles = self.board.moveTileNumbers(tilenumbers[:], m)
                score, critical = self.evaluate(newtiles)
                newtiles = self.board.setCell(newtiles, critical[0], critical[1], 2)
                if depth != 0:
                    my_m, my_s = self.nextMoveRecur(newtiles[:], depth - 1, maxDepth)
                    score += my_s * pow(base, maxDepth - depth + 1)
                if (score > bestScore):
                    bestMove = m
                    bestScore = score

        return bestMove, bestScore

    def evaluate(self, tilenumbers, commonRatio=0.25):

        maxVal = 0.
        criticalTile = (-1, -1)

        for i in range(8):
            linearWeightedVal = 0
            invert = False if i < 4 else True
            weight = 1.
            ctile = (-1, -1)

            cond = i % 4
            for y in range(4):
                for x in range(4):
                    if cond == 0:
                        b_x = 4 - 1 - x if invert else x
                        b_y = y
                    elif cond == 1:
                        b_x = x
                        b_y = 4 - 1 - y if invert else y
                    elif cond == 2:
                        b_x = 4 - 1 - x if invert else x
                        b_y = 4 - 1 - y
                    elif cond == 3:
                        b_x = 4 - 1 - x
                        b_y = 4 - 1 - y if invert else y

                    currVal = self.board.getCell(tilenumbers, b_x, b_y)
                    if (currVal == 0 and ctile == (-1, -1)):
                        ctile = (b_x, b_y)
                    linearWeightedVal += currVal * weight
                    weight *= commonRatio
                invert = not invert

            if linearWeightedVal > maxVal:
                maxVal = linearWeightedVal
                criticalTile = ctile

        return maxVal, criticalTile

    def solveBoard(self, moveinterval=500):
        boardHWND = self.board.getWindowHandle()
        if not boardHWND:
            return False
        bx, by, bw, bh = cv2.boundingRect(self.board.getBoardContour())
        x0, x1, y0, y1 = bx, bx + bw, by, by + bh

        win32gui.SetForegroundWindow(boardHWND)
        shell = win32com.client.Dispatch('WScript.Shell')
        print
        'Set the focus to the Game Window, and the press this arrow key:'
        keymove = ['UP', 'DOWN', 'LEFT', 'RIGHT']

        delay = moveinterval / 3  # milliseconds delay to cancel board animation effect
        prev_numbers = []
        while True:
            numbers, inframe, outframe = self.board.update()
            if numbers != prev_numbers:
                cv2.waitKey(delay)
                numbers, inframe, outframe = self.board.update()
                if numbers == prev_numbers:  # recheck if has changed
                    continue
                prev_numbers = numbers
                move = ai.nextMove()
                if move:
                    key = keymove[move - 1]
                    shell.SendKeys('{%s}' % key)
                    print
                    key
                    cv2.waitKey(delay)
                    cv2.imshow('CV copy', inframe[y0:y1, x0:x1])
                    cv2.imshow('CV out', outframe[y0:y1, x0:x1])
            cv2.waitKey(delay)
        cv2.destroyWindow('CV copy')
        cv2.destroyWindow('CV out')


# http://gabrielecirulli.github.io/2048/
# http://ov3y.github.io/2048-AI/
board = Board("2048 - Google Chrome")
# board = Board("2048 - Mozilla Firefox")

ai = AI(board)
ai.solveBoard(360)

print()
'stopped.'