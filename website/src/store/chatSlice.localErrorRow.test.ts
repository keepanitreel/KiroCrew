/**
 * An `error` row written only in this browser (an attachment failure reported
 * to a chat that was off screen) has no server copy. Reopening that chat must
 * not replace it away with the fetched history page.
 */
import { describe, it, expect } from 'vitest'
import reducer, { appendSlotMessage, switchSlot } from '../store/chatSlice'
import { LOCAL_ERROR_META } from './chat/transcript'

const SLOT = 'slot-a'
const TS = '2026-01-01T00:00:00.000Z'
const page = {
  key: SLOT, messages: [{ role: 'assistant', content: 'hi', cls: '', ts: TS, meta: { mid: 'm-hi' } }],
  hasMore: false, running: false, total: 1, queue: [],
}

function reopenAfter(meta: Record<string, unknown> | undefined) {
  let state = reducer(undefined, { type: '@@INIT' })
  state = {
    ...state,
    activeSlot: 'slot-b',
    slotMessages: { [SLOT]: [{ role: 'assistant', content: 'hi', cls: '', ts: TS, meta: { mid: 'm-hi' } }] },
  }
  state = reducer(state, appendSlotMessage({ slot: SLOT, message: { role: 'error', content: 'Upload failed: disk full', cls: '', meta } }))
  state = reducer(state, switchSlot.pending('req', SLOT))
  return reducer(state, switchSlot.fulfilled(page as never, 'req', SLOT))
}

describe('a browser-only error row across a slot switch', () => {
  it('keeps the row when the chat is reopened and its history lands', () => {
    const state = reopenAfter({ [LOCAL_ERROR_META]: true })
    expect(state.messages.map(m => m.role)).toEqual(['assistant', 'error'])
    expect(state.messages[1].content).toBe('Upload failed: disk full')
  })

  it('still lets the page replace an unmarked error row', () => {
    const state = reopenAfter(undefined)
    expect(state.messages.map(m => m.role)).toEqual(['assistant'])
  })
})
