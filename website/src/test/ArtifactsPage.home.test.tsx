/**
 * The Artifacts page on a desktop: the side list (Home, All artifacts, the
 * folder tree) and Home's two groups.
 *
 * Home splits on the server's own `sweepable` flag: Recent is everything the
 * sweep can never delete, Recent output is what it could, plus unsaved session
 * documents. These tests pin that split, the cap wording, keeping an output
 * by its star, and the side list driving the URL.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { screen, waitFor, within, cleanup } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import ArtifactsPage from '../pages/ArtifactsPage'
import { renderWithProviders } from './helpers'
import { api } from '../api/client'
import type { Artifact, ArtifactFolder, SessionDoc } from '../types'

vi.mock('../api/client')
vi.mock('../hooks/useIsMobile', () => ({ useIsMobile: () => false }))
vi.mock('@virtuoso.dev/masonry', () => ({ VirtuosoMasonry: () => <div data-testid="masonry" /> }))

const mkArtifact = (slug: string, overrides: Partial<Artifact> = {}): Artifact => ({
  slug,
  name: slug.replace(/-/g, ' '),
  kind: 'widget',
  source: 'chat',
  pinned: false,
  description: '',
  tags: [],
  version: 1,
  created_at: '2026-11-01T10:00:00+00:00',
  updated_at: '2026-11-01T10:00:00+00:00',
  ...overrides,
})

const mkFolder = (id: string, name: string, overrides: Partial<ArtifactFolder> = {}): ArtifactFolder => ({
  id, name, order: 0, parent_id: '', item_count: 0, ...overrides,
})

const doc: SessionDoc = {
  path: '/work/plan.md',
  name: 'plan.md',
  updated_at: '2026-11-02T10:00:00',
  session_key: 'dashboard_chat-1',
  session_title: 'Doc indicators',
  message_ts: 'm1',
  saved: false,
  slug: '',
}

function seed({ artifacts = [] as Artifact[], folders = [] as ArtifactFolder[], docs = [] as SessionDoc[], cap = 200 as number | null } = {}) {
  const m = vi.mocked(api)
  m.artifacts = vi.fn().mockResolvedValue({ artifacts, ...(cap === null ? {} : { output_cap: cap }) })
  m.artifactFolders = vi.fn().mockResolvedValue({ folders })
  m.artifactSessionDocs = vi.fn().mockResolvedValue({ docs })
  m.getArtifactPublishProviders = vi.fn().mockResolvedValue({ providers: [], kind: 'widget' })
  m.publishProviders = vi.fn().mockResolvedValue({ providers: [] })
  m.themeBoot = vi.fn().mockResolvedValue({})
  m.artifact = vi.fn().mockImplementation((slug: string) =>
    Promise.resolve(artifacts.find((a) => a.slug === slug) ?? mkArtifact(slug)))
  m.setArtifactPinned = vi.fn().mockResolvedValue({})
  return m
}

const home = () => screen.getByTestId('artifacts-home')
const group = (title: string) => within(home()).getByRole('heading', { name: title }).closest('section') as HTMLElement

beforeEach(() => {
  vi.clearAllMocks()
  localStorage.clear()
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false }))
})
afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

describe('Artifacts Home', () => {
  it('splits made-or-kept from output on the server sweep flag', async () => {
    seed({
      artifacts: [
        mkArtifact('red-main-audit', { kind: 'markdown' }),
        mkArtifact('starred-widget', { auto_registered: true, pinned: true, sweepable: false }),
        mkArtifact('chat-widget', { auto_registered: true, sweepable: true }),
      ],
      docs: [doc],
    })
    renderWithProviders(<ArtifactsPage />)

    await waitFor(() => expect(within(group('Recent')).getByText('red main audit')).toBeInTheDocument())
    const recent = group('Recent')
    expect(within(recent).getByText('starred widget')).toBeInTheDocument()
    expect(within(recent).queryByText('chat widget')).toBeNull()

    const output = group('Recent output')
    expect(within(output).getByText('chat widget')).toBeInTheDocument()
    // An unsaved session document is output too, though it is no library record.
    expect(within(output).getByText('plan.md')).toBeInTheDocument()
    expect(within(output).queryByText('red main audit')).toBeNull()
  })

  it('says how much output there is against the sweep cap', async () => {
    seed({ artifacts: [mkArtifact('a-one', { sweepable: true }), mkArtifact('a-two', { sweepable: true })], cap: 200 })
    renderWithProviders(<ArtifactsPage />)
    await waitFor(() => expect(within(home()).getByText(/2 of 200 before the oldest clear/)).toBeInTheDocument())
  })

  it('falls back to wording with no number when the gateway sends no cap', async () => {
    seed({ artifacts: [mkArtifact('a-one', { sweepable: true })], cap: null })
    renderWithProviders(<ArtifactsPage />)
    await waitFor(() => expect(within(home()).getByText(/clears as newer output arrives/)).toBeInTheDocument())
  })

  it('keeps an output by starring it, without opening it', async () => {
    const user = userEvent.setup()
    const m = seed({ artifacts: [mkArtifact('chat-widget', { auto_registered: true, sweepable: true })] })
    renderWithProviders(<ArtifactsPage />)
    const output = await waitFor(() => group('Recent output'))
    await user.click(within(output).getByRole('button', { name: 'Star to keep' }))
    expect(m.setArtifactPinned).toHaveBeenCalledWith('chat-widget', true)
  })

  it('puts the two groups side by side in the table view', async () => {
    localStorage.setItem('mc-artifacts-view', 'table')
    seed({ artifacts: [mkArtifact('kept-doc', { kind: 'markdown' }), mkArtifact('chat-widget', { sweepable: true })] })
    renderWithProviders(<ArtifactsPage />)
    await waitFor(() => expect(within(group('Recent')).getByText('kept doc')).toBeInTheDocument())
    expect(home().className).toMatch(/lg:grid-cols-2/)
    expect(within(group('Recent output')).getByText('chat widget')).toBeInTheDocument()
  })

  it('offers no Starred filter on Home, since Home already splits kept from not', async () => {
    seed({ artifacts: [mkArtifact('kept-doc')] })
    renderWithProviders(<ArtifactsPage />)
    await waitFor(() => expect(home()).toBeInTheDocument())
    expect(screen.queryByRole('group', { name: /starred/i })).toBeNull()
  })
})

describe('Artifacts side list', () => {
  it('lists the folder tree with direct counts and opens a folder', async () => {
    const user = userEvent.setup()
    seed({
      artifacts: [mkArtifact('filed-one', { folder_id: 'ops' }), mkArtifact('loose-one')],
      folders: [mkFolder('ops', 'Ops')],
    })
    renderWithProviders(<ArtifactsPage />)
    const nav = await waitFor(() => screen.getByRole('navigation', { name: 'Artifact library' }))
    const ops = await waitFor(() => within(nav).getByRole('button', { name: /^Ops/ }))
    expect(ops.textContent).toContain('1')

    await user.click(ops)
    await waitFor(() => expect(screen.getByRole('heading', { name: 'Ops' })).toBeInTheDocument())
    expect(screen.queryByTestId('artifacts-home')).toBeNull()
  })

  it('shows every artifact flat under All artifacts, filed or not', async () => {
    const user = userEvent.setup()
    seed({
      artifacts: [mkArtifact('filed-one', { folder_id: 'ops' }), mkArtifact('loose-one')],
      folders: [mkFolder('ops', 'Ops')],
    })
    localStorage.setItem('mc-artifacts-view', 'table')
    renderWithProviders(<ArtifactsPage />)
    const nav = await waitFor(() => screen.getByRole('navigation', { name: 'Artifact library' }))
    await user.click(within(nav).getByRole('button', { name: /All Artifacts/ }))
    await waitFor(() => expect(screen.getByRole('heading', { name: 'All Artifacts' })).toBeInTheDocument())
    expect(screen.getByText('loose one')).toBeInTheDocument()
  })
})
