"""The git/tar self-target floor: a git or tar command that dials this host is refused.

The ssh self-target floor (``argv_floor_ssh``) reads the ssh-family verbs ``ssh``,
``scp``, ``sftp`` and ``rsync``.  Programs that start an ssh transport on their own are
not those verbs, so an agent can re-enter this same host, outside the sandbox mounts,
through them: ``git clone localhost:x`` scp-dials the local sshd, ``git fetch
ssh://127.0.0.1/x`` the same over a URL, ``git clone ext::ssh localhost id`` runs a
command that dials it, ``git -c core.sshCommand='ssh localhost' clone far:x`` redirects
every ssh to this host, and ``tar -cf localhost:out.tar dir`` rsh-dials it.  This floor
reads a git or tar command line -- behind env/wrapper unwrapping, quoting, ``user@``
prefixes, URI schemes, globs and the same shell expansions the ssh floor resolves -- and
refuses it when a transport operand names this host (the same ``argv_floor_hosts``
self-host test the ssh floor uses) or when a command the transport runs DIALS this host
(``git ext::``, a ``core.sshCommand`` / ``GIT_SSH`` / ``GIT_SSH_COMMAND`` override, ``tar
--rsh-command``, an ``--upload-pack`` / ``--receive-pack`` program, an invoked alias);
``TAR_OPTIONS`` and ``TAPE`` are read as tar arguments.  Every such command must name a
literal self host and is then read by the same argv-floor dispatcher ``is_denied`` runs
on a whole command (``_reenter_floors``), so ``ssh localhost`` is refused by the ssh
floor and ``git ls-remote localhost:x`` by this one; an invoked alias is rebuilt into
the invocation git runs and re-entered the same way.
Such a command runs in the agent's own sandbox, so one that does not dial this host is
no escape and stays allowed.  It FAILS CLOSED on a transport argument it cannot parse.

A git/tar remote host, and a host named in such a command value, is refused only when it
is a LITERAL self address (loopback names and addresses, this host's own cached
names/addresses), with no DNS or hosts-file alias lookup, so a first-seen dotted remote
such as ``github.com`` or a ProxyJump bastion is never pending-refused.  A remote host is
read as typed: git and tar never glob-expand it, so ``*:*`` or ``local*:x`` names no self
host (globs are read in the PROGRAM word, ``g*t``, and inside a command value, which a
shell runs and expands: ``GIT_SSH_COMMAND='ssh local*'`` is still refused).  A dotted name that resolves to
THIS host through an alias is the documented residual for git/tar: the ssh floor still
covers ssh/scp, and the OS network fence is the fix of record.

Deliberately NOT refused: normal local use -- ``git add``/``commit``/``status``/``log``/
``diff``, ``git clone /local/path``, ``git clone https://…`` (http(s)/git/ftp is not an
ssh-or-exec escape), ``git fetch origin`` with a remote that is not this host, ``tar
-czf x.tgz dir``, ``tar -xf x.tar``, and file-operand globs (``git add s* *.*``, ``tar
-czf handoff.tgz s* *.*``).

``kiro_crew.security.argv_floor`` serves every name defined here, and a patch through it
reaches this module (:mod:`kiro_crew._facade_mirror`).  It reads a command line through
the ssh floor's own front end -- its source pre-walk (``_ssh_floor_source_text``), its
nested-payload frames and its environment tracker (``_FrameEnvironment``) -- and uses
the ssh floor's operand resolver, the host verdicts in ``argv_floor_hosts``, and the
shell reader.  A new shell normalization goes in the ssh floor, which both floors read;
a new git/tar transport spelling goes here.
"""

from __future__ import annotations

import shlex
from contextvars import ContextVar
from typing import TYPE_CHECKING
from urllib.parse import unquote

from ..trust_patterns import ENV_ASSIGNMENT_RE
from . import shell_normalizer as _shell_normalizer  # noqa: F401  (facade patch target parity)
from .argv_floor_hosts import _host_is_self
from .argv_floor_self import _SELF_FLOOR_MACHINERY_RE, _SELF_FLOOR_QUOTE_JUNK_RE
from .argv_floor_ssh import (
    _FUNCTION_BIND_CAP,
    _command_value_names_self_endpoint,
    _FrameEnvironment,
    _operand_targets_self,
    _ssh_floor_source_text,
    _unmask_separators,
)
from .shell_normalizer import (
    _decode_shell_quoted_literals,
    _ends_argv,
    _glob_could_expand_to,
    _nested_shell_payloads,
    _self_tokens,
    _shell_payload_walk,
    _strip_redirect,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

# ── git/tar sandbox-escape floor ──
# A git or tar command can open an ssh transport (or run an arbitrary command)
# that re-enters this host outside the sandbox, without ever naming an
# ssh-family verb the ssh floor reads.  The forms, all resolved below:
#   * scp-like ``host:path`` and the ssh/``git+ssh``/``ssh+git`` URI schemes
#     (git) and ``host:file`` (tar) whose host is THIS machine;
#   * a command git or tar runs in place of the transport: a decoded
#     ``git clone ext::<command>`` payload, a ``core.sshCommand`` config
#     (``-c``) or ``GIT_SSH`` / ``GIT_SSH_COMMAND`` / ``GIT_PROXY_COMMAND``
#     environment override, ``tar --rsh-command=<command>``, an
#     ``--upload-pack`` / ``--receive-pack`` / ``--exec`` (fetch-family ``-u``)
#     program and an invoked ``!`` alias.  Every one goes through
#     ``_command_value_dials_self``: a literal self host with no DNS, then the
#     whole argv-floor dispatcher re-entered on the command
#     (``_reenter_floors``).  An invoked ordinary alias is rebuilt into the
#     git invocation it becomes and re-entered the same way.  Any such command
#     runs inside the agent's own sandbox, so one that does not dial this host
#     is not an escape.
# Everything else -- local paths, bare words (git reads a bare operand as a
# local path, never ssh), http(s)/git/ftp URLs, remotes that are not this
# host, archive member globs -- stays allowed.  The git/tar verb is located behind
# ``VAR=x`` and wrappers (``sudo -u root``, ``timeout 5``) by ``argv_rules``'s reader.


# Subcommands that open a transport themselves.  ``archive``, ``submodule`` and
# ``remote`` need a second-level classification and are handled below.
_GIT_TRANSPORT_VERBS: frozenset[str] = frozenset(
    {"clone", "fetch", "pull", "push", "ls-remote", "fetch-pack", "send-pack", "bundle"}
)
# Facade manifest names: transport discovery does not read either set to
# choose which tokens are visible.
_GIT_REMOTE_FIRST_OPERAND_VERBS: frozenset[str] = frozenset(
    {"fetch", "pull", "ls-remote", "push", "bundle", "send-pack", "fetch-pack"}
)
_GIT_REMOTE_ANY_OPERAND_VERBS: frozenset[str] = frozenset({"clone", "submodule", "remote"})


# ``git remote`` subcommands that query a remote now, rather than only editing
# the local remote configuration.
_GIT_REMOTE_CONNECTING_SUBCOMMANDS: frozenset[str] = frozenset({"show", "prune", "update"})


# The config key (lower-cased) that hands git an arbitrary ssh-replacement
# command, and the environment variables that choose a transport command.
_GIT_SSHCOMMAND_CONFIG_PREFIX = "core.sshcommand="
_GIT_SSH_ENV_NAMES: frozenset[str] = frozenset({"git_proxy_command", "git_ssh", "git_ssh_command"})


# git schemes that are NOT an ssh-or-exec escape: the connection is a plain
# socket to a server, not a login shell on this host.
_GIT_NON_ESCAPE_SCHEMES: tuple[str, ...] = (
    "http://",
    "https://",
    "git://",
    "ftp://",
    "ftps://",
    "file://",
)


# git ssh-transport scheme aliases normalized to ``ssh://`` so the ssh floor's
# operand resolver reads their authority (it models ``ssh://`` but not these).
_GIT_SSH_SCHEME_ALIASES: tuple[str, ...] = ("git+ssh://", "ssh+git://")


def _unmasked(token: str) -> str:
    """*token* with the floor's quoted-separator sentinels restored, de-quoted."""
    return _unmask_separators(token).strip().strip("\"'")


def _decode_url_authority(url: str) -> str:
    """*url* with ``%HH`` escapes resolved in the authority only.

    git percent-decodes the host of a URL authority before dialing it
    (``ssh://local%68ost/x`` reaches ``localhost``), so the literal self check
    must read the decoded host.  Only the authority -- between ``//`` and the
    next ``/`` -- is decoded; a ``%`` in the path is left alone.
    """
    marker = url.find("://")
    if marker < 0:
        return url
    start = marker + len("://")
    end = url.find("/", start)
    if end < 0:
        end = len(url)
    authority = url[start:end]
    if "%" not in authority:
        return url
    return url[:start] + unquote(authority) + url[end:]


def _decode_scp_host(operand: str) -> str:
    """*operand* (``[user@]host:path``) with ``%HH`` resolved in the host part only."""
    if operand.startswith("["):
        close = operand.find("]")
        if close > 0:
            return "[" + unquote(operand[1:close]) + "]" + operand[close + 1 :]
    head, sep, rest = operand.partition(":")
    if not sep or "%" not in head:
        return operand
    return unquote(head) + sep + rest


def _frame_commands(source: str) -> "list[str]":
    """*source*'s tokens with separators real at THIS frame's nesting level.

    A nested shell payload's own ``;`` / ``&&`` was masked as DATA by the
    top-level source pre-walk, because it was quoted one frame up; at the
    payload's own level an unquoted separator is a real command boundary.
    Git's sq-quote apostrophe escape is masked while the shell reader removes
    the outer assignment quotes, then restored for the config decoder.
    Restoring separators (``_unmask_separators``) and re-running the ssh floor's
    source pre-walk (``_ssh_floor_source_text``) re-masks only the separators
    quoted in THIS frame, so ``_split_commands`` sees the same boundaries the
    ssh floor's own token scan sees.  Idempotent for the top-level frame, whose
    quoted separators stay masked.
    """
    source = _unmask_separators(source).replace("'\\''", "\ue100")
    return [
        token.replace("\ue100", "'\\''") for token in _self_tokens(_ssh_floor_source_text(source))
    ]


def _git_remote_targets_self(operand: str) -> bool:
    """True if a git remote *operand* dials THIS host over an ssh transport.

    Only a transport-shaped operand can dial: a URL (``scheme://``) or a
    scp-like ``host:path`` (a colon before the first slash, non-empty host).
    http(s)/git/ftp/file URLs are not an ssh-or-exec escape and return False;
    the ``git+ssh``/``ssh+git`` aliases are normalized to ``ssh://`` so the ssh
    floor's resolver reads their authority.  A BARE word -- a branch name, a
    number, ``origin``, a local path, even ``localhost`` -- is a local path to
    git, never ssh, so it returns False without any host or DNS lookup.  The
    host is read LITERALLY: ``dns=False`` (literal self addresses only; see the
    module docstring), so ``git@github.com:x`` is never pending-refused, and
    ``match_globs=False``, because git hands its host to ssh as typed and
    never glob-expands it -- ``*:*`` and ``local*:x`` name no self host.
    """
    t = operand.strip().strip("\"'")
    if not t:
        return False
    low = t.lower()
    if "://" in t:
        for scheme in _GIT_NON_ESCAPE_SCHEMES:
            if low.startswith(scheme):
                return False
        for alias in _GIT_SSH_SCHEME_ALIASES:
            if low.startswith(alias):
                t = "ssh://" + t[len(alias) :]
                break
        t = _decode_url_authority(t)
        return _operand_targets_self(t, host_position=True, dns=False, match_globs=False)
    head = t.split("/", 1)[0]
    if ":" not in head or not head.split(":", 1)[0]:
        return False  # a bare word or local path: git never dials it
    t = _decode_scp_host(t)
    if t.startswith("["):
        close = t.find("]")
        if close > 0 and t[close + 1 :].startswith(":"):
            host = t[1:close]
            if "@" in host:
                host = host.rsplit("@", 1)[1]
            before_port, sep, _port = host.rpartition(":")
            if sep and before_port and ":" not in before_port:
                host = before_port
            if _operand_targets_self(host, host_position=True, dns=False, match_globs=False):
                return True
    return _operand_targets_self(t, host_position=True, dns=False, match_globs=False)


# A value word longer than this is not split into every glued-option suffix:
# the literal reader answers "may be self" and the ssh floor's own reading of
# the value decides.  No real ssh option bundle comes near it.
_VALUE_WORD_SUFFIX_CAP = 256


def _value_names_literal_self(value: str) -> bool:
    """True if a command *value* names a LITERAL self host anywhere in it.

    A WIDE literal read with no DNS: the value is read through the ssh floor's
    source pre-walk (``_ssh_floor_source_text``: static substitutions, same-line
    assignments), its ANSI-C quoting is resolved and its quotes removed, then
    every word -- and, for an option word, every glued suffix (``-Jlocalhost``,
    ``-4vJlocalhost``) -- is checked by the ssh floor's literal endpoint reader
    (``_operand_targets_self(..., dns=False)`` plus its ``:`` / ``=`` / ``,``
    fragments).  A word, or an option word's ``=`` value, is also read as a git
    remote (``_git_remote_targets_self``, literal too), so a carried
    ``git+ssh://``, bracketed or ``%``-encoded self remote is seen.  It only
    narrows: ``_command_value_dials_self`` still needs a floor to refuse the
    value.
    """
    text = _unmask_separators(_ssh_floor_source_text(value))
    text = _decode_shell_quoted_literals(text)
    text = text.replace('"', "").replace("'", "").replace("\\", "")
    # An ``ext::`` payload carried inside the value embeds a space as ``% ``;
    # reading it as a plain space keeps the host before it a word of its own.
    text = text.replace("% ", " ")
    for word in text.split():
        if not word.startswith("-"):
            if _command_value_names_self_endpoint(word) or _git_remote_targets_self(word):
                return True
            continue
        bare = word.lstrip("-")
        if len(bare) > _VALUE_WORD_SUFFIX_CAP:
            return True
        _, sep, attached = bare.partition("=")
        if sep and _git_remote_targets_self(attached):
            return True
        for start in range(len(bare)):
            if _command_value_names_self_endpoint(bare[start:]):
                return True
    return False


#: Where the floor read in progress sits inside commands carried by other commands:
#: ``(depth, alias names expanded, command texts re-entered)`` along the current
#: chain only.  ``_reenter_floors`` threads it, since the dispatcher takes a string.
_REENTRY_CHAIN: "ContextVar[tuple[int, frozenset[str], frozenset[str]]]" = ContextVar(
    "git_tar_floor_reentry_chain", default=(0, frozenset(), frozenset())
)


def _reenter_floors(command: str, *, alias: "str | None" = None) -> bool:
    """True if the argv-floor dispatcher refuses *command*, which this git/tar command runs.

    *command* is a whole command line a git or tar invocation hands to a shell or
    runs as its transport (a command value), or the git invocation an invoked
    ordinary alias becomes.  It is read by the SAME dispatcher ``is_denied`` runs
    on an agent's whole command (``deny_gate._reentered_floor_hit``: every
    self-protection argv floor, row-backed ones only while the operator's row set
    enables them, the hit reported under the inner row's own id), so a floor added
    there reads carried commands too.  Re-entry is bounded by the ssh
    floor's named-body expansion limit, ``_FUNCTION_BIND_CAP``: a chain deeper
    than that, an alias name already expanded on this chain (*alias*) or a
    command text already re-entered on it is a loop git itself would reject, and
    is refused (fail closed), as is a dispatcher that raises on the command.
    """
    depth, aliases, commands = _REENTRY_CHAIN.get()
    if depth >= _FUNCTION_BIND_CAP or command in commands:
        return True
    if alias is not None:
        if alias in aliases:
            return True
        aliases = aliases | {alias}
    from .deny_gate import _reentered_floor_hit

    chain = _REENTRY_CHAIN.set((depth + 1, aliases, commands | {command}))
    try:
        return _reentered_floor_hit(command) is not None
    except Exception:  # incl. RecursionError: an unreadable carried command
        return True
    finally:
        _REENTRY_CHAIN.reset(chain)


def _command_value_dials_self(value: str, *, alias: "str | None" = None) -> bool:
    """True if a command *value* git or tar runs dials THIS host.

    Every command-carrying value comes here: ``GIT_SSH`` / ``GIT_SSH_COMMAND`` /
    ``GIT_PROXY_COMMAND``, any invocation-local config value (``core.sshCommand``,
    ``core.gitProxy`` and the rest), a decoded ``ext::`` payload, an
    ``--upload-pack`` / ``--receive-pack`` / ``--exec`` / ``-u`` program, an
    invoked ``!`` alias (*alias* names it) and ``tar --rsh-command``.  It is
    refused when it names a literal self host (``_value_names_literal_self``, no
    DNS) AND the whole argv-floor dispatcher refuses it (``_reenter_floors``), so
    ``git ls-remote localhost:x`` is read by this floor and ``ssh localhost`` by
    the ssh floor.  The literal read runs first, so a value naming only remote
    hosts -- ``ssh -o ProxyJump=bastion.example.com`` -- never reaches the ssh
    floor's DNS layer and is never pending-refused.
    """
    return _value_names_literal_self(value) and _reenter_floors(value, alias=alias)


def _git_ext_command(operand: str, following: "list[str]") -> "str | None":
    """The command an ``ext::<command>`` remote *operand* runs, else None.

    An unquoted ``ext::ssh localhost id`` reaches git as several operands, so
    the *following* operands are appended -- reading more words only widens
    what ``_command_value_dials_self`` sees.  Git separates arguments on
    unescaped spaces; ``% `` embeds a space in one argument and ``%%`` embeds a
    percent sign.
    """
    t = operand.strip().strip("\"'")
    if not t.lower().startswith("ext::"):
        return None
    encoded = " ".join([t[len("ext::") :], *following])
    arguments = [""]
    index = 0
    while index < len(encoded):
        if encoded.startswith("% ", index):
            arguments[-1] += " "
            index += 2
        elif encoded.startswith("%%", index):
            arguments[-1] += "%"
            index += 2
        elif encoded[index] == " ":
            arguments.append("")
            index += 1
        else:
            arguments[-1] += encoded[index]
            index += 1
    return " ".join(arguments)


def _git_remote_candidates(token: str) -> "tuple[str, ...]":
    """Transport operands or executable overrides carried by one post-verb *token*.

    The whole token covers ordinary operands and detached option values: every
    post-verb token is a candidate, so ``--opt value`` has its value read
    whatever the option.  The attached ``--opt=value`` therefore has its value
    read for every long option too (``--repo=``, ``--remote=``,
    ``--upload-pack=``), and fetch's glued short ``-u<program>``.  Config
    options are also classified separately with their key intact.
    """
    t = token.strip().strip("\"'")
    candidates = [t]
    if t.startswith("--"):
        _, sep, attached = t.partition("=")
        if sep:
            candidates.append(attached)
    if t.startswith("-u") and len(t) > 2:
        short_tail = t[2:]
        candidates.append(short_tail[1:] if short_tail.startswith("=") else short_tail)
    return tuple(candidate for candidate in candidates if candidate)


def _git_config_override(tokens: "list[str]", index: int) -> "tuple[str, str, int] | None":
    """Return ``(option, config, consumed)`` for a git config option at *index*."""
    tok = tokens[index]
    if tok in ("-c", "--config", "--config-env"):
        if index + 1 >= len(tokens):
            return None
        return tok, tokens[index + 1], 2
    if tok.startswith("--config-env="):
        return "--config-env", tok[len("--config-env=") :], 1
    if tok.startswith("--config="):
        return "--config", tok[len("--config=") :], 1
    if tok.startswith("-c") and len(tok) > 2:
        return "-c", tok[2:], 1
    return None


def _git_nested_subcommand(tokens: "list[str]") -> "str | None":
    """First non-option word below ``git remote`` or ``git submodule``."""
    options_ended = False
    for tok in tokens:
        if _ends_argv(tok) and tok in (";", "&", "|", "\n", "&&", "||"):
            break
        if not options_ended and tok == "--":
            options_ended = True
            continue
        if options_ended or not tok.startswith("-"):
            return tok.lower()
    return None


def _git_options_before_terminator(tokens: "list[str]") -> "tuple[str, ...]":
    """Option tokens before this git subcommand's genuine ``--`` terminator."""
    options: "list[str]" = []
    for tok in tokens:
        if _ends_argv(tok) and tok in (";", "&", "|", "\n", "&&", "||"):
            break
        if tok == "--":
            break
        if tok.startswith("-"):
            options.append(tok)
    return tuple(options)


def _git_transport_modes(verb: str, tokens: "list[str]") -> "tuple[bool, bool]":
    """Return ``(opens_transport, inspect_remote_tokens)`` for this git verb."""
    if verb in _GIT_TRANSPORT_VERBS:
        return True, True
    options = _git_options_before_terminator(tokens)
    if verb == "archive":
        opens = any(tok == "--remote" or tok.startswith("--remote=") for tok in options)
        return opens, opens
    if verb == "submodule":
        opens = _git_nested_subcommand(tokens) in ("add", "update")
        return opens, opens
    if verb != "remote":
        return False, False

    subcommand = _git_nested_subcommand(tokens)
    if subcommand == "add":
        opens = any(tok in ("-f", "--fetch") for tok in options)
        return opens, True  # records a URL even without ``--fetch``
    if subcommand == "set-url":
        records = "--delete" not in options
        return False, records
    if subcommand in _GIT_REMOTE_CONNECTING_SUBCOMMANDS:
        opens = subcommand != "show" or not any(tok in ("-n", "--no-query") for tok in options)
        return opens, opens
    if subcommand == "set-head":
        opens = any(tok in ("-a", "--auto") for tok in options)
        return opens, opens
    return False, False


def _git_tokens_target_self(tokens: "list[str]") -> bool:
    """True if any post-verb transport operand or executable targets self."""
    for idx, tok in enumerate(tokens):
        if _ends_argv(tok) and tok in (";", "&", "|", "\n", "&&", "||"):
            break
        for masked_candidate in _git_remote_candidates(tok):
            candidate = _unmasked(masked_candidate)
            if _git_remote_targets_self(candidate) or _command_value_dials_self(candidate):
                return True
            ext_command = _git_ext_command(masked_candidate, tokens[idx + 1 :])
            if ext_command is not None and _command_value_dials_self(_unmasked(ext_command)):
                return True
    return False


def _tar_archive_targets_self(operand: str) -> bool:
    """True if a tar archive *operand* (``host:file``) rsh-dials THIS host.

    tar reads an archive name with a colon before any slash as ``host:file``
    (optionally ``user@host:file``) and reaches it over its remote shell, so a
    self host is a self dial.  A local name (no such colon) returns False.
    """
    t = operand.strip().strip("\"'")
    if not t:
        return False
    t = _decode_scp_host(t)
    head = t.split("/", 1)[0]
    if ":" not in head:
        return False
    host = head.split(":", 1)[0]
    if "@" in host:
        host = host.rsplit("@", 1)[1]
    if not host:
        return False
    # Read literally, like a git remote host: tar hands the host to its remote
    # shell as typed, so a glob host names no self host.
    return _host_is_self(host, dns_fallback=False, match_globs=False) or _operand_targets_self(
        t, host_position=True, dns=False, match_globs=False
    )


def _env_assignment(stripped_tok: str) -> "tuple[str, str] | None":
    """``(name, value)`` of any ``NAME=value`` word, the name lowered, else None.

    The parser ``_FrameEnvironment`` reads git's environment with: every
    variable counts (``GIT_SSH*``, the numbered ``GIT_CONFIG_*`` config, and
    any name a ``--config-env`` points at).
    """
    if not ENV_ASSIGNMENT_RE.match(stripped_tok):
        return None
    name, _, value = stripped_tok.partition("=")
    return name.lower(), value


def _is_env_name(name: str) -> bool:
    """True if a bare ``export`` operand *name* is a variable name."""
    return ENV_ASSIGNMENT_RE.match(name + "=") is not None


def _observe_words(frame_env: "_FrameEnvironment", words: "Iterable[str]") -> None:
    """Feed each of *words* (already de-quoted) to *frame_env*."""
    for word in words:
        frame_env.observe(word, word)


def _git_command_targets_self(tokens: "list[str]", env: "dict[str, list[str]]") -> bool:
    """True if a ``git`` command's *tokens* (its args) dial THIS host, directly or not."""
    i = 0
    n = len(tokens)
    global_overrides: "list[tuple[str, str]]" = []
    # Locate the subcommand first.  Global ssh-command overrides are collected
    # here but checked only after the subcommand is known to open a transport.
    # A redirection is not a word of git's argv (bash removes it before exec),
    # so it is stepped over the way the ssh floor steps over it: a bare or
    # fd-prefixed operator (``2>/dev/null``, ``>&2``, ``<file``, ``&>x``) is
    # skipped, and a detached operator (``>`` ``file``, ``2>`` ``file``) also
    # skips its target.  ``clone>/dev/null`` keeps its word part as the verb.
    redirect_target_pending = False
    verb_word = ""
    while i < n:
        tok = tokens[i]
        if redirect_target_pending:
            redirect_target_pending = False
            i += 1
            continue
        if ">" in tok or "<" in tok:
            remainder = _strip_redirect(tok)
            if not remainder or remainder.isdigit() or remainder == "&":
                redirect_target_pending = tok.endswith((">", "<"))
                i += 1
                continue
            if not remainder.startswith("-"):
                verb_word = remainder
                break
        if not tok.startswith("-"):
            verb_word = tok
            break
        config_override = _git_config_override(tokens, i)
        if config_override is not None:
            option, config, consumed = config_override
            global_overrides.append((option, config))
            i += consumed
            continue
        if tok in (
            "-C",
            "--git-dir",
            "--work-tree",
            "--namespace",
            "--super-prefix",
            "--attr-source",
        ):
            i += 2  # option + its value
            continue
        i += 1
    if i >= n:
        return False

    verb = _unmasked(verb_word).lower()
    command_tokens = tokens[i + 1 :]
    config_overrides = list(global_overrides)
    idx = 0
    while idx < len(command_tokens):
        tok = command_tokens[idx]
        if tok == "--":
            break
        config_override = _git_config_override(command_tokens, idx)
        if config_override is None:
            idx += 1
            continue
        option, config, consumed = config_override
        config_overrides.append((option, config))
        idx += consumed

    # The numbered environment config (``GIT_CONFIG_COUNT`` with
    # ``GIT_CONFIG_KEY_<n>`` / ``GIT_CONFIG_VALUE_<n>``) is the same config
    # as ``-c <key>=<value>``.  Every key present is read, a superset of
    # the ones the count enables.
    for name, keys in env.items():
        if name.startswith("git_config_key_"):
            values = env.get("git_config_value_" + name[len("git_config_key_") :]) or [""]
            config_overrides.extend(("-c", key + "=" + value) for key in keys for value in values)

    for parameters in env.get("git_config_parameters", ()):
        try:
            entries = shlex.split(_unmask_separators(parameters), comments=False, posix=True)
        except ValueError:
            # shortcut: malformed sq-quoting falls back to whitespace before the classifier.
            entries = _unmask_separators(parameters).split()
        config_overrides.extend(("-c", entry) for entry in entries)

    opens_transport, inspect_remotes = _git_transport_modes(verb, command_tokens)
    if not inspect_remotes:
        return _git_alias_dials_self(
            verb, tokens[:i], command_tokens, config_overrides, global_overrides, env
        )

    if opens_transport:
        # A transport-command environment override matters only when this
        # invocation can run a transport.  Local verbs merely store/read data.
        for name in _GIT_SSH_ENV_NAMES:
            if any(_command_value_dials_self(_unmasked(value)) for value in env.get(name, ())):
                return True

    for option, config in config_overrides:
        if _sshcommand_config_dials_self(config, env, option):
            return True

    # Remote and command discovery is token-agnostic: no option grammar
    # decision can hide a self-targeting operand or executable override.
    return _git_tokens_target_self(command_tokens)


def _config_entries(
    option: str, config: str, env: "dict[str, list[str]]"
) -> "list[tuple[str, str]]":
    """The ``(key, value)`` entries one invocation-local config *option* supplies.

    For ``--config-env`` the value names a variable in the command environment,
    and every value kept under that name is an entry.
    """
    key, sep, value = _unmasked(config).partition("=")
    if not sep:
        return []
    if option == "--config-env":
        return [(key, _unmasked(carried)) for carried in env.get(value.lower()) or [""]]
    return [(key, value)]


def _sshcommand_config_dials_self(config: str, env: "dict[str, list[str]]", option: str) -> bool:
    """True if an invocation-local git config entry dials this host.

    Every value is checked as a command git could execute
    (``_command_value_dials_self``).  Only transport-URL keys are also checked as
    remotes, including the rewrite base carried in a ``url.<base>.insteadOf`` or
    ``url.<base>.pushInsteadOf`` key.
    """
    for key, value in _config_entries(option, config, env):
        # The command-value check is KEY-AGNOSTIC: any config value git runs as
        # a command (``core.sshCommand``, ``core.gitProxy`` and the rest) is read
        # as one.
        if _command_value_dials_self(value):
            return True
        # The remote/URL check is scoped to a TRANSPORT-URL key only.  An
        # http-proxy value (``http.proxy=localhost:3128``) is a plain
        # ``host:port`` for a proxy, not a dialed transport, so a key-agnostic
        # remote check over-refused it.
        low_key = key.lower()
        if _is_transport_url_key(low_key) and _git_remote_targets_self(value):
            return True
        if not low_key.startswith("url."):
            continue
        for suffix in (".insteadof", ".pushinsteadof"):
            if low_key.endswith(suffix) and _git_remote_targets_self(
                key[len("url.") : -len(suffix)]
            ):
                return True
    return False


def _exported_environment(env: "dict[str, list[str]]", extra: "Iterable[str]" = ()) -> str:
    """Shell text exporting every value in *env* (and the *extra* assignments).

    A rebuilt command is re-entered as text, so the environment the invocation
    runs with leads it as ``export NAME=value …;``: every value kept under a
    name is exported, the way ``_FrameEnvironment`` keeps them all.
    """
    assignments = [
        name + "=" + shlex.quote(_unmask_separators(value))
        for name, values in env.items()
        for value in values
    ]
    assignments.extend(extra)
    return "export " + " ".join(assignments) + "; " if assignments else ""


def _git_alias_dials_self(
    verb: str,
    prefix: "list[str]",
    operands: "list[str]",
    config_overrides: "list[tuple[str, str]]",
    global_overrides: "list[tuple[str, str]]",
    env: "dict[str, list[str]]",
) -> bool:
    """True if *verb* invokes an invocation-local alias whose command dials this host.

    git replaces the alias word with its definition and keeps the rest of the
    invocation, so the command is rebuilt whole and re-entered
    (``_reenter_floors``): its environment, then for an ordinary alias
    ``git``, every word before the verb (*prefix*: ``-c``, ``--config-env``,
    ``-C`` …), the alias's git arguments and the invocation's *operands*; for a
    ``!`` alias, its shell command followed by the *operands*, run with git's
    ``-c`` config in ``GIT_CONFIG_PARAMETERS`` the way git hands it to the
    child.  A nested alias is expanded by the re-entered read itself, and
    re-expanding one already on the chain is refused as a loop.
    """
    bodies = [
        value
        for option, config in config_overrides
        for key, value in _config_entries(option, config, env)
        if key.lower() == "alias." + verb
    ]
    words = [shlex.quote(_unmask_separators(tok)) for tok in operands]
    for body in bodies:
        if body.startswith("!"):
            parameters = " ".join(
                shlex.quote(key + "=" + value)
                for option, config in global_overrides
                for key, value in _config_entries(option, config, env)
            )
            extra = ["git_config_parameters=" + shlex.quote(parameters)] if parameters else []
            command = _exported_environment(env, extra) + " ".join([body[1:], *words])
            if _command_value_dials_self(command, alias=verb):
                return True
            continue
        try:
            alias_args = shlex.split(body, comments=False, posix=True)
        except ValueError:
            alias_args = body.split()
        if not alias_args:
            continue
        rebuilt = _exported_environment(env) + " ".join(
            [
                "git",
                *(shlex.quote(_unmask_separators(tok)) for tok in prefix),
                *(shlex.quote(arg) for arg in alias_args),
                *words,
            ]
        )
        if _reenter_floors(rebuilt, alias=verb):
            return True
    return False


def _is_transport_url_key(low_key: str) -> bool:
    """True if a lower-cased git config *low_key* carries a transport URL value.

    The keys whose VALUE is a remote git can dial: ``remote.<name>.url``,
    ``remote.<name>.pushurl`` and the ``url.<base>.insteadOf`` /
    ``url.<base>.pushInsteadOf`` rewrites (checked for their value here; the
    rewrite's base is checked by the key-suffix branch in
    ``_sshcommand_config_dials_self``).
    """
    if low_key.startswith("remote.") and (low_key.endswith(".url") or low_key.endswith(".pushurl")):
        return True
    return low_key.startswith("url.") and (
        low_key.endswith(".insteadof") or low_key.endswith(".pushinsteadof")
    )


def _tar_command_targets_self(tokens: "list[str]", env: "dict[str, list[str]]") -> bool:
    """True if a ``tar`` command's effective argv rsh-dials THIS host."""
    tar_options_values = env.get("tar_options", [])
    if len(tar_options_values) > 1:
        return any(
            _tar_command_targets_self(tokens, {**env, "tar_options": [tar_options]})
            for tar_options in tar_options_values
        )
    tar_options = tar_options_values[0] if tar_options_values else None
    try:
        default_tokens = (
            shlex.split(_unmask_separators(tar_options), comments=False, posix=True)
            if tar_options is not None
            else []
        )
    except ValueError:
        # shortcut: malformed quoting falls back to whitespace before the same classifier.
        default_tokens = _unmask_separators(tar_options or "").split()

    effective_tokens = [*default_tokens, *tokens]
    archive_candidates: "list[str]" = []
    options_ended = False
    for tok in effective_tokens:
        if _ends_argv(tok) and tok in (";", "&", "|", "\n", "&&", "||"):
            break
        # A remote-shell command may be a detached token or an attached long
        # option value.  Check both without deciding which option owns it.
        command_candidates = [tok]
        if "=" in tok:
            command_candidates.append(tok.split("=", 1)[1])
        if any(_command_value_dials_self(_unmasked(candidate)) for candidate in command_candidates):
            return True
        low = tok.lower()
        if not options_ended and tok == "--":
            options_ended = True
            continue
        if not options_ended and tok.startswith("--"):
            if low.startswith("--file="):
                archive_candidates.append(tok.split("=", 1)[1])
            continue
        if not options_ended and tok.startswith("-"):
            letters = tok[1:]
            if "f" in letters:
                # A glued ``-f<archive>`` names the archive in this same word.
                after = letters.split("f", 1)[1]
                if after:
                    archive_candidates.append(after[1:] if after.startswith("=") else after)
            continue
        # Check every non-option word, not only the archive slot inferred from
        # tar's option grammar.  That includes detached ``-f`` values, every
        # old-style bundle operand and every member after ``--``; ordinary
        # member names and option values are rejected by the narrow host:file
        # predicate itself.
        archive_candidates.append(tok)
    # The ``TAPE`` default archive is read whenever it is present: a self-shaped
    # TAPE is refused even next to an explicit ``-f`` archive, since the narrow
    # host:file predicate -- not tar's option grammar -- decides, and a non-self
    # TAPE (``/dev/st0``) is simply not refused.
    archive_candidates.extend(env.get("tape", ()))
    return any(_tar_archive_targets_self(_unmasked(archive)) for archive in archive_candidates)


def _is_git_tar_to_self(text_lower: str) -> bool:
    """True if *text_lower* starts a git/tar network transport to THIS host.

    The ssh pre-walk and payload reader expose hidden transports and inherited environments.
    FAIL-CLOSED: an unparseable transport is refused. If payload discovery raises,
    analysis degrades to the top-level frame; a carried command that names a literal
    self host is still re-entered (``_command_value_dials_self``) and refused when
    a floor refuses it or cannot read it.
    """
    from .argv_rules import _program_args, _split_commands
    from .shell_commands import read_commands

    text_lower = _ssh_floor_source_text(text_lower)
    # A quick gate that is a SUPERSET of every spelling the walk below can read
    # as git/tar: a literal ``git``/``tar``, or any quoting or expansion
    # machinery that lets the shell build the verb from other text -- quote and
    # backslash splices (``gi''t``, ``"git"``), globs and braces (``g*t``,
    # ``/usr/bin/g?t``), ``$`` and backticks.  The character classes are the
    # self floor's own (``argv_floor_self``), so nothing new is matched here.
    if (
        "git" not in text_lower
        and "tar" not in text_lower
        and not _SELF_FLOOR_MACHINERY_RE.search(text_lower)
        and not _SELF_FLOOR_QUOTE_JUNK_RE.search(text_lower)
    ):
        return False

    def _payload_key(source: str) -> "tuple[tuple[str, ...], ...]":
        return tuple(tuple(command) for command in _split_commands(_frame_commands(source)))

    # A payload frame runs with the environment of the parent command that
    # launched it.  Keying by the payload's argv lets duplicate payload text
    # retain every launch environment without letting a later command rewrite
    # an earlier launch.
    inherited_by_payload: "dict[tuple[tuple[str, ...], ...], list[dict[str, list[str]]]]" = {}
    try:
        payload_frames = _shell_payload_walk(text_lower)
    except Exception:
        payload_frames = [(text_lower, [])]
    for frame_index, (source, _tokens) in enumerate(payload_frames):
        inherited_envs = (
            [{}] if frame_index == 0 else inherited_by_payload.get(_payload_key(source), [{}])
        )
        frame_env = _FrameEnvironment(_env_assignment, _is_env_name)
        # Redirection-free argv comes from the shared quote-aware shell reader,
        # so only live redirects and their detached targets disappear. Scan it
        # before the original frame; the latter preserves every existing
        # expansion, separator and nested-payload reading.
        redirect_free_sources = [
            " ".join(source[word.start : word.end] for word in shell_command.argv)
            for shell_command in read_commands(source, nested=False)
            if shell_command.redirects
            and shell_command.argv
            and shell_command.argv[-1].start > shell_command.redirects[0][0].end
        ]
        raw_commands = [
            command
            for view in (*redirect_free_sources, source)
            for command in _split_commands(_frame_commands(view))
        ]
        for index, raw_command in enumerate(raw_commands):
            if index:
                # ``_split_commands`` drops the separator ``_FrameEnvironment``
                # clears a command's pending assignments on, so feed it back.
                frame_env.observe(";", ";")
            # A separator QUOTED at this frame's level stays a sentinel (data),
            # so the walks below do not stop at it; only the value actually
            # inspected is unmasked (``_unmasked``), as the ssh floor unmasks
            # only the operand it checks.
            command = [
                t.strip() if _env_assignment(t.strip()) is not None else t.strip().strip("\"'")
                for t in raw_command
            ]
            parsed = _program_args(command)
            if parsed is None:
                _observe_words(frame_env, command)
                continue
            verb, args = parsed
            prefix = command[: len(command) - len(args) - 1]
            _observe_words(frame_env, prefix)
            # Every value kept under a name is carried, inherited ones included:
            # which case the shell honors is not known (``_FrameEnvironment``).
            envs: "list[dict[str, list[str]]]" = []
            for inherited in inherited_envs:
                env = {name: list(values) for name, values in inherited.items()}
                for layer in (frame_env.exported, frame_env.pending):
                    for name, values in layer.items():
                        env.setdefault(name, []).extend(values)
                envs.append(env)
            try:
                nested_payloads = _nested_shell_payloads(command)
            except Exception:
                nested_payloads = []
            for payload in nested_payloads:
                key = _payload_key(payload)
                if key:
                    inherited_by_payload.setdefault(key, []).extend(envs)
            _observe_words(frame_env, command[len(prefix) :])
            verb = _unmasked(verb)
            if verb.endswith(".exe"):
                verb = verb[: -len(".exe")]
            # A glob program word (``g*t``, ``/usr/bin/g?t``) is expanded by the
            # shell before exec, so it is read as every verb it CAN name, the
            # way the self floor reads ``pkill`` spellings.  One glob can name
            # both, so the two checks are independent.
            for env in envs:
                if verb == "git" or _glob_could_expand_to(verb, ("git",)):
                    if _git_command_targets_self(args, env):
                        return True
                if verb == "tar" or _glob_could_expand_to(verb, ("tar",)):
                    if _tar_command_targets_self(args, env):
                        return True
    return False
