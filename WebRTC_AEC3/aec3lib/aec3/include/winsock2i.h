#ifndef _WINSOCK2API_
// Prevent inclusion of winsock.h#ifdef _WINSOCKAPI_#error Header winsock.h is included unexpectedly.#endif
// NOTE: If you use Windows Platform SDK, you should enable following definition:// #define USING_WIN_PSDK
#if !defined(WIN32_LEAN_AND_MEAN) && (_WIN32_WINNT >= 0x0400) && !defined(USING_WIN_PSDK)
#include <windows.h>
#else#include <winsock2.h>
#endif
#endif//_WINSOCK2API_
